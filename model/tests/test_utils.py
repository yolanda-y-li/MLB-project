"""
Shared utilities for the extended model test suite.

Loads the trained checkpoint, rebuilds the graph (with the same seed used at
training time) and produces (probs, preds, labels, pairs) for any split, plus
small helpers used by the individual test scripts.

If a trained checkpoint is not available, the helpers can also train a fresh
model from scratch with the same default config so the tests can still run.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Allow running scripts from anywhere
THIS_DIR = Path(__file__).parent
MODEL_DIR = THIS_DIR.parent
PROJECT_DIR = MODEL_DIR.parent
RESULTS_DIR = THIS_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(MODEL_DIR))

from data_loader import load_data, NUM_CLASSES, LABEL_MAP, NO_INTERACTION  # noqa: E402
from model import GeneDrugRGCN  # noqa: E402

CLASS_NAMES = ["binds", "upregulates", "downregulates", "no_interaction"]
INTERACTION_LABELS = [LABEL_MAP[k] for k in ("binds", "upregulates", "downregulates")]


# ---------------------------------------------------------------------------
# Default config (mirrors model/main.py defaults — keep in sync)
# ---------------------------------------------------------------------------
DEFAULT_CFG = dict(
    hidden_dim=128,
    num_layers=3,
    dropout=0.05,
    lr=5e-3,
    weight_decay=1e-5,
    epochs=100,
    batch_size=4096,
    neg_strategy="edge_swap",
    neg_ratio=1.0,
    patience=15,
    seed=42,
)


def get_device(force: str | None = None) -> torch.device:
    if force:
        return torch.device(force)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_checkpoint() -> Path | None:
    ckpt = MODEL_DIR / "checkpoints" / "best_model.pt"
    if ckpt.exists():
        return ckpt
    return None


# ---------------------------------------------------------------------------
# Build / load model
# ---------------------------------------------------------------------------

def build_model(info: dict, cfg: dict, metadata) -> GeneDrugRGCN:
    return GeneDrugRGCN(
        n_genes=info["n_genes"],
        n_drugs=info["n_drugs"],
        n_classes=info["n_classes"],
        n_families=info["n_families"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        metadata=metadata,
    )


def load_or_train(cfg: dict | None = None, device: torch.device | None = None,
                  verbose: bool = True):
    """
    Returns (model, train_graph, splits, info, used_cfg).
    Loads checkpoint if present, otherwise trains a fresh model with cfg.
    """
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    device = device or get_device()

    if verbose:
        print(f"[test_utils] Device: {device}")
        print(f"[test_utils] Loading data with seed={cfg['seed']}, "
              f"neg_strategy={cfg['neg_strategy']}, neg_ratio={cfg['neg_ratio']}")

    train_graph, splits, info = load_data(
        neg_strategy=cfg["neg_strategy"],
        neg_ratio=cfg["neg_ratio"],
        seed=cfg["seed"],
    )
    train_graph = train_graph.to(device)

    model = build_model(info, cfg, train_graph.metadata()).to(device)

    # SAGEConv(-1, -1) lazy init
    with torch.no_grad():
        model.encode(train_graph.edge_index_dict)

    ckpt = find_checkpoint()
    if ckpt is not None:
        if verbose:
            print(f"[test_utils] Loading checkpoint from {ckpt}")
        state = torch.load(ckpt, map_location=device)
        # `state` is the dict saved by main.py
        model_state = state.get("model_state", state)
        try:
            model.load_state_dict({k: v.to(device) for k, v in model_state.items()})
        except RuntimeError as exc:
            if verbose:
                print(f"[test_utils] Checkpoint shape mismatch ({exc}); retraining from scratch.")
            ckpt = None

    if ckpt is None:
        if verbose:
            print("[test_utils] No usable checkpoint — training from scratch.")
        _train_in_place(model, train_graph, splits, cfg, device, verbose)

    model.eval()
    return model, train_graph, splits, info, cfg


def _train_in_place(model, train_graph, splits, cfg, device, verbose):
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"],
                           weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=7, factor=0.5
    )
    criterion = nn.CrossEntropyLoss()

    train_pairs, train_labels = [t.to(device) for t in splits["train"]]
    val_pairs, val_labels = [t.to(device) for t in splits["val"]]

    edge_index_dict = train_graph.edge_index_dict
    bs = cfg["batch_size"]
    best_val = -1.0
    best_state = None
    no_improve = 0

    from sklearn.metrics import roc_auc_score

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        x_dict = model.encode(edge_index_dict)
        optimizer.zero_grad()
        n = len(train_labels)
        for start in range(0, n, bs):
            end = min(start + bs, n)
            bp = train_pairs[start:end]
            bl = train_labels[start:end]
            logits = model.classifier(torch.cat(
                [x_dict["compound"][bp[:, 0]], x_dict["gene"][bp[:, 1]]], dim=-1))
            loss = criterion(logits, bl)
            (loss * (end - start) / n).backward(retain_graph=True)
        optimizer.step()

        # Quick val AUROC
        model.eval()
        with torch.no_grad():
            x_dict = model.encode(edge_index_dict)
            logits = model.classifier(torch.cat(
                [x_dict["compound"][val_pairs[:, 0]],
                 x_dict["gene"][val_pairs[:, 1]]], dim=-1))
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            try:
                auroc = roc_auc_score(val_labels.cpu().numpy(), probs,
                                      multi_class="ovr", average="macro")
            except ValueError:
                auroc = 0.0
        scheduler.step(auroc)
        if auroc > best_val:
            best_val = auroc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"  [train] epoch {epoch:3d}  val AUROC {auroc:.4f}  best {best_val:.4f}")
        if no_improve >= cfg["patience"]:
            if verbose:
                print(f"  [train] early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Save so subsequent test scripts pick it up
    ckpt_dir = MODEL_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": {k: v.cpu() for k, v in model.state_dict().items()},
         "args": cfg,
         "info": {k: v for k, v in {
             "n_genes": train_graph["gene"].num_nodes,
             "n_drugs": train_graph["compound"].num_nodes,
             "n_classes": train_graph["pharmacologic_class"].num_nodes,
             "n_families": train_graph["gene_family"].num_nodes,
         }.items()}},
        ckpt_dir / "best_model.pt",
    )
    if verbose:
        print(f"  [train] saved checkpoint -> {ckpt_dir / 'best_model.pt'}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_split(model, train_graph, pairs: torch.Tensor, batch_size: int = 4096,
                  device: torch.device | None = None) -> np.ndarray:
    """Returns probs [N, NUM_CLASSES]."""
    device = device or next(model.parameters()).device
    pairs = pairs.to(device)
    model.eval()
    x_dict = model.encode(train_graph.edge_index_dict)
    out = []
    n = len(pairs)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        bp = pairs[start:end]
        logits = model.classifier(torch.cat(
            [x_dict["compound"][bp[:, 0]], x_dict["gene"][bp[:, 1]]], dim=-1))
        out.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


def get_split_predictions(model, train_graph, splits, split_name: str = "test",
                          batch_size: int = 4096):
    pairs, labels = splits[split_name]
    probs = predict_split(model, train_graph, pairs, batch_size=batch_size)
    preds = probs.argmax(axis=-1)
    return {
        "pairs": pairs.cpu().numpy(),
        "labels": labels.cpu().numpy(),
        "probs": probs,
        "preds": preds,
    }

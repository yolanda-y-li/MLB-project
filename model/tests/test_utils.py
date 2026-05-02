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
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def _log(msg: str):
    """Always-flush logger so progress is visible during long runs."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

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
    """Look in a few common spots so a hand-placed best_model.pt is found."""
    candidates = [
        MODEL_DIR / "checkpoints" / "best_model.pt",
        MODEL_DIR / "best_model.pt",
        PROJECT_DIR / "best_model.pt",
        PROJECT_DIR / "checkpoints" / "best_model.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
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


# --- shared in-process caches: every test reuses the same model + encoded
# graph + split predictions, so we don't re-load data and re-encode once
# per test (which on CPU was the dominant cost). -----------------------------
_LOAD_CACHE: dict = {}
_X_DICT_CACHE: dict = {}
_PRED_CACHE: dict = {}


def load_or_train(cfg: dict | None = None, device: torch.device | None = None,
                  verbose: bool = True):
    """
    Returns (model, train_graph, splits, info, used_cfg).
    Loads checkpoint if present, otherwise trains a fresh model with cfg.
    Cached after the first call within the same Python process.
    """
    if "result" in _LOAD_CACHE:
        return _LOAD_CACHE["result"]
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    device = device or get_device()

    # Peek at checkpoint first so we can build the model with the right
    # architecture hyperparams (whoever trained it may have used different
    # hidden_dim / num_layers / dropout than today's defaults).
    ckpt = find_checkpoint()
    ckpt_args = None
    if ckpt is not None:
        try:
            peek = torch.load(ckpt, map_location="cpu")
            ckpt_args = peek.get("args") if isinstance(peek, dict) else None
        except Exception as exc:
            if verbose:
                print(f"[test_utils] Could not peek at checkpoint args: {exc}")

    if ckpt_args:
        for key in ("hidden_dim", "num_layers", "dropout",
                    "neg_strategy", "neg_ratio", "seed"):
            if key in ckpt_args and ckpt_args[key] is not None:
                cfg[key] = ckpt_args[key]
        if verbose:
            _log(f"Using config from checkpoint: "
                 f"hidden_dim={cfg['hidden_dim']} num_layers={cfg['num_layers']} "
                 f"dropout={cfg['dropout']} seed={cfg['seed']}")

    if verbose:
        _log(f"Device: {device}")
        _log(f"Loading data with seed={cfg['seed']}, "
             f"neg_strategy={cfg['neg_strategy']}, neg_ratio={cfg['neg_ratio']}")

    t0 = time.time()
    train_graph, splits, info = load_data(
        neg_strategy=cfg["neg_strategy"],
        neg_ratio=cfg["neg_ratio"],
        seed=cfg["seed"],
    )
    if verbose:
        _log(f"Data loaded in {time.time() - t0:.1f}s — "
             f"genes={info['n_genes']:,} drugs={info['n_drugs']:,} "
             f"families={info['n_families']:,} "
             f"train_pairs={len(splits['train'][0]):,} "
             f"test_pairs={len(splits['test'][0]):,}")
    train_graph = train_graph.to(device)

    if verbose:
        _log("Building model …")
    t0 = time.time()
    model = build_model(info, cfg, train_graph.metadata()).to(device)
    if verbose:
        _log(f"Model built in {time.time() - t0:.1f}s")

    # SAGEConv(-1, -1) lazy init — this is the first full graph encode and
    # is usually the single most expensive step on CPU.
    if verbose:
        _log("Lazy-initializing SAGEConv weights with one full graph encode "
             "(this is the slow step on CPU) …")
    t0 = time.time()
    with torch.no_grad():
        model.encode(train_graph.edge_index_dict)
    if verbose:
        _log(f"Lazy init / first encode finished in {time.time() - t0:.1f}s")

    if ckpt is not None:
        if verbose:
            _log(f"Loading checkpoint from {ckpt}")
        state = torch.load(ckpt, map_location=device)
        model_state = state.get("model_state", state)
        try:
            model.load_state_dict({k: v.to(device) for k, v in model_state.items()})
            if verbose:
                _log("Checkpoint loaded OK")
        except RuntimeError as exc:
            if verbose:
                _log(f"Checkpoint shape mismatch ({exc}); retraining from scratch.")
            ckpt = None

    if ckpt is None:
        if verbose:
            _log("No usable checkpoint — training from scratch.")
        _train_in_place(model, train_graph, splits, cfg, device, verbose)

    model.eval()
    _LOAD_CACHE["result"] = (model, train_graph, splits, info, cfg)
    return _LOAD_CACHE["result"]


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
def get_x_dict(model, train_graph):
    """Encode the full graph once; subsequent calls reuse the cached x_dict."""
    if "x_dict" in _X_DICT_CACHE:
        return _X_DICT_CACHE["x_dict"]
    _log("Encoding full graph (one-time, cached for the rest of the run) …")
    t0 = time.time()
    model.eval()
    x_dict = model.encode(train_graph.edge_index_dict)
    _X_DICT_CACHE["x_dict"] = x_dict
    _log(f"Graph encoded in {time.time() - t0:.1f}s")
    return x_dict


@torch.no_grad()
def predict_split(model, train_graph, pairs: torch.Tensor, batch_size: int = 4096,
                  device: torch.device | None = None) -> np.ndarray:
    """Returns probs [N, NUM_CLASSES]. The expensive graph encode is cached."""
    device = device or next(model.parameters()).device
    pairs = pairs.to(device)
    x_dict = get_x_dict(model, train_graph)
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
    """Cached per (split_name) so we don't re-predict each test."""
    if split_name in _PRED_CACHE:
        return _PRED_CACHE[split_name]
    pairs, labels = splits[split_name]
    probs = predict_split(model, train_graph, pairs, batch_size=batch_size)
    preds = probs.argmax(axis=-1)
    result = {
        "pairs": pairs.cpu().numpy(),
        "labels": labels.cpu().numpy(),
        "probs": probs,
        "preds": preds,
    }
    _PRED_CACHE[split_name] = result
    return result

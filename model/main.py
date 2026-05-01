"""
Training and evaluation entry point for the Gene-Drug Interaction R-GCN.

Usage:
    python main.py [options]

Key options:
    --hidden_dim    embedding / hidden size (default 128)
    --num_layers    R-GCN depth (default 3)
    --dropout       dropout rate (default 0.05)
    --lr            learning rate (default 0.005)
    --epochs        number of training epochs (default 100)
    --neg_strategy  "edge_swap" (default) or "random"
    --device        "cuda" / "cpu" (auto-detected if omitted)

Metrics reported: AUROC, Average Precision, macro-F1 (per split).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)

# Allow running from project root or from model/ directory
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_data, NUM_CLASSES
from model import GeneDrugRGCN


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    probs    = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    y_true   = labels.detach().cpu().numpy()
    y_pred   = probs.argmax(axis=-1)
    n_cls    = NUM_CLASSES

    # One-hot encode for AP calculation
    y_onehot = np.eye(n_cls)[y_true]

    try:
        auroc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    except ValueError:
        auroc = float("nan")

    try:
        ap = average_precision_score(y_onehot, probs, average="macro")
    except ValueError:
        ap = float("nan")

    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"auroc": auroc, "ap": ap, "f1": f1}


# ---------------------------------------------------------------------------
# Train / eval loops  (full-batch: GNN runs once, classifier batches)
# ---------------------------------------------------------------------------

def run_epoch(model, edge_index_dict, pairs, labels, optimizer, criterion,
              training: bool, batch_size: int, device):
    """One full pass over pairs.

    When training=True:
      - GNN encoding is done once on the full graph (retaining the
        computation graph), then the classifier is applied in mini-batches.
      - Gradients accumulate, and optimizer.step() is called once at the end.
    When training=False:
      - Same batching but inside torch.no_grad().
    """
    model.train(training)
    context = torch.enable_grad() if training else torch.no_grad()

    pairs  = pairs.to(device)
    labels = labels.to(device)

    with context:
        # Full-graph encoding (once per epoch)
        x_dict = model.encode(edge_index_dict)

        if training:
            optimizer.zero_grad()

        all_logits, all_labels = [], []
        total_loss = 0.0
        n = len(labels)

        for start in range(0, n, batch_size):
            end          = min(start + batch_size, n)
            batch_pairs  = pairs[start:end]
            batch_labels = labels[start:end]

            drug_emb = x_dict["compound"][batch_pairs[:, 0]]
            gene_emb = x_dict["gene"][batch_pairs[:, 1]]
            logits   = model.classifier(torch.cat([drug_emb, gene_emb], dim=-1))

            loss = criterion(logits, batch_labels)

            if training:
                # Scale loss so gradients are averaged across all batches
                (loss * (end - start) / n).backward(retain_graph=True)

            total_loss  += loss.item() * (end - start)
            all_logits.append(logits.detach().cpu())
            all_labels.append(batch_labels.cpu())

        if training:
            optimizer.step()

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_logits, all_labels)
    metrics["loss"] = total_loss / n
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Gene-Drug Interaction R-GCN")
    p.add_argument("--hidden_dim",    type=int,   default=128)
    p.add_argument("--num_layers",    type=int,   default=3)
    p.add_argument("--dropout",       type=float, default=0.05)
    p.add_argument("--lr",            type=float, default=5e-3)
    p.add_argument("--weight_decay",  type=float, default=1e-5)
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch_size",    type=int,   default=4096)
    p.add_argument("--neg_strategy",  type=str,   default="edge_swap",
                   choices=["edge_swap", "random"])
    p.add_argument("--neg_ratio",     type=float, default=1.0,
                   help="negatives per positive example")
    p.add_argument("--patience",      type=int,   default=15,
                   help="early stopping patience (epochs without val AUROC improvement)")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--device",        type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save",          type=str,
                   default=str(Path(__file__).parent / "checkpoints" / "best_model.pt"))
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)
    print(f"Device : {device}")
    print(f"Config : {vars(args)}\n")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("Loading data …")
    train_graph, splits, info = load_data(
        neg_strategy=args.neg_strategy,
        neg_ratio=args.neg_ratio,
        seed=args.seed,
    )
    train_graph = train_graph.to(device)
    edge_index_dict = train_graph.edge_index_dict

    train_pairs, train_labels = splits["train"]
    val_pairs,   val_labels   = splits["val"]
    test_pairs,  test_labels  = splits["test"]

    print(
        f"Pairs  — train: {len(train_labels):,}  "
        f"val: {len(val_labels):,}  "
        f"test: {len(test_labels):,}"
    )
    print(
        f"Nodes  — gene: {info['n_genes']:,}  "
        f"compound: {info['n_drugs']:,}  "
        f"pharm_class: {info['n_classes']:,}  "
        f"gene_family: {info['n_families']:,}"
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = GeneDrugRGCN(
        n_genes    = info["n_genes"],
        n_drugs    = info["n_drugs"],
        n_classes  = info["n_classes"],
        n_families = info["n_families"],
        hidden_dim = args.hidden_dim,
        num_layers = args.num_layers,
        dropout    = args.dropout,
        metadata   = train_graph.metadata(),
    ).to(device)

    # SAGEConv(-1, -1) is lazily initialized — materialize weights before use
    with torch.no_grad():
        model.encode(edge_index_dict)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params — {n_params:,}\n")

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=7, factor=0.5
    )
    criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # Training loop with early stopping
    # ------------------------------------------------------------------
    best_val_auroc = -1.0
    best_state     = None
    no_improve     = 0

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(
            model, edge_index_dict, train_pairs, train_labels,
            optimizer, criterion, training=True,
            batch_size=args.batch_size, device=device,
        )
        vl = run_epoch(
            model, edge_index_dict, val_pairs, val_labels,
            optimizer, criterion, training=False,
            batch_size=args.batch_size, device=device,
        )
        scheduler.step(vl["auroc"])

        if vl["auroc"] > best_val_auroc:
            best_val_auroc = vl["auroc"]
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve     = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | "
                f"Train loss {tr['loss']:.4f}  AUROC {tr['auroc']:.4f}  F1 {tr['f1']:.4f} | "
                f"Val   loss {vl['loss']:.4f}  AUROC {vl['auroc']:.4f}  F1 {vl['f1']:.4f}"
            )

        if no_improve >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (no val AUROC improvement for {args.patience} epochs).")
            break

    # ------------------------------------------------------------------
    # Test evaluation on best checkpoint
    # ------------------------------------------------------------------
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    te = run_epoch(
        model, edge_index_dict, test_pairs, test_labels,
        optimizer, criterion, training=False,
        batch_size=args.batch_size, device=device,
    )

    print("\n=== Test Results ===")
    print(f"  AUROC : {te['auroc']:.4f}")
    print(f"  AP    : {te['ap']:.4f}")
    print(f"  F1    : {te['f1']:.4f}")
    print(f"  Loss  : {te['loss']:.4f}")

    # ------------------------------------------------------------------
    # Save checkpoint
    # ------------------------------------------------------------------
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state":   best_state,
            "args":          vars(args),
            "info":          {k: v for k, v in info.items()
                              if not isinstance(v, dict)},
            "test_metrics":  te,
        },
        save_path,
    )
    print(f"\nCheckpoint saved -> {save_path}")


if __name__ == "__main__":
    main()

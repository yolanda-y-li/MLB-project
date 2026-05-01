"""
Negative-sampling robustness.

The model was trained with edge-swap negatives by default. Are these
negatives "too easy" or "too hard"? We re-evaluate the trained model under
two different negative regimes:

  * EDGE_SWAP — same as training (hard negatives)
  * RANDOM    — random (drug, gene) pairs not in the positive set (easy)

If the model is robust, performance should be comparable. A big drop on
random negatives would suggest the model is overfit to the edge-swap
distribution; a big drop on edge-swap would suggest it can't separate
hard cases. Either way it's a useful diagnostic.

We do NOT retrain — we just regenerate the test split's negatives under
the alternative strategy and re-score the model.

Output:
  results/negative_robustness.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from test_utils import (
    NUM_CLASSES,
    NO_INTERACTION,
    LABEL_MAP,
    RESULTS_DIR,
    DEFAULT_CFG,
    load_or_train,
    predict_split,
)

# We need access to the internal negative samplers in data_loader
sys.path.insert(0, str(Path(__file__).parent.parent))
from data_loader import (  # noqa: E402
    _edge_swap_negatives,
    _random_negatives,
    EDGE_CODE_TO_REL,
)
import pandas as pd  # noqa: E402
import random  # noqa: E402

DATA_DIR = Path(__file__).parent.parent.parent / "Data"


def _load_test_positives_and_set(seed: int):
    """Recreate the same test-positive split + global positive set."""
    gene_df = pd.read_csv(DATA_DIR / "nodes/gene_nodes.tsv", sep="\t",
                          header=None, names=["id", "symbol", "type"])
    drug_df = pd.read_csv(DATA_DIR / "nodes/drug_nodes.tsv", sep="\t",
                          header=None, names=["id", "name", "type"])
    gene_id2idx = {gid: i for i, gid in enumerate(gene_df["id"])}
    drug_id2idx = {did: i for i, did in enumerate(drug_df["id"])}
    n_genes = len(gene_df)
    n_drugs = len(drug_df)

    gd = pd.read_csv(DATA_DIR / "edges/gene_drug_edges.tsv", sep="\t",
                     header=None, names=["compound", "edge_code", "gene"])
    gd["rel"] = gd["edge_code"].map(EDGE_CODE_TO_REL)
    gd["label"] = gd["rel"].map(LABEL_MAP)
    gd["drug_idx"] = gd["compound"].map(drug_id2idx)
    gd["gene_idx"] = gd["gene"].map(gene_id2idx)
    gd = gd.dropna(subset=["drug_idx", "gene_idx", "label"]).copy()
    gd = gd.astype({"drug_idx": int, "gene_idx": int, "label": int})
    gd = gd.sample(frac=1, random_state=seed).reset_index(drop=True)

    n_total = len(gd)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    test_pos = gd.iloc[n_train + n_val:].copy()
    all_positive_set = set(zip(gd["drug_idx"], gd["gene_idx"]))
    return test_pos, all_positive_set, n_drugs, n_genes


def _build_pairs(test_pos, neg_pairs):
    pos_pairs = list(zip(test_pos["drug_idx"], test_pos["gene_idx"]))
    pos_labels = list(test_pos["label"])
    pairs = pos_pairs + neg_pairs
    labels = pos_labels + [NO_INTERACTION] * len(neg_pairs)
    return (
        torch.tensor(pairs, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def _metrics(probs, labels):
    pred = probs.argmax(axis=1)
    out = {}
    try:
        out["auroc_macro"] = roc_auc_score(labels, probs, multi_class="ovr",
                                           average="macro",
                                           labels=list(range(NUM_CLASSES)))
    except ValueError:
        out["auroc_macro"] = float("nan")
    try:
        out["ap_macro"] = average_precision_score(np.eye(NUM_CLASSES)[labels],
                                                  probs, average="macro")
    except ValueError:
        out["ap_macro"] = float("nan")
    out["f1_macro"] = f1_score(labels, pred, average="macro", zero_division=0,
                               labels=list(range(NUM_CLASSES)))
    out["accuracy"] = float((pred == labels).mean())
    return out


def run(out_path: Path | None = None):
    out_path = out_path or (RESULTS_DIR / "negative_robustness.csv")
    model, graph, splits, info, cfg = load_or_train()

    seed = cfg["seed"]
    test_pos, all_pos_set, n_drugs, n_genes = _load_test_positives_and_set(seed)
    n_neg = max(1, int(len(test_pos) * cfg["neg_ratio"]))

    rows = []
    for strategy in ("edge_swap", "random"):
        random.seed(seed + (1 if strategy == "random" else 0))
        np.random.seed(seed + (1 if strategy == "random" else 0))
        if strategy == "edge_swap":
            neg = _edge_swap_negatives(test_pos, n_neg, all_pos_set, n_drugs, n_genes)
        else:
            neg = _random_negatives(n_neg, all_pos_set, n_drugs, n_genes)
        pairs, labels = _build_pairs(test_pos, neg)
        probs = predict_split(model, graph, pairs)
        m = _metrics(probs, labels.numpy())
        rows.append({
            "negative_strategy": strategy,
            "n_pos": int(len(test_pos)),
            "n_neg": int(len(neg)),
            **{k: round(v, 6) if not np.isnan(v) else "" for k, v in m.items()},
        })

    fieldnames = ["negative_strategy", "n_pos", "n_neg",
                  "auroc_macro", "ap_macro", "f1_macro", "accuracy"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"[negative_robustness] wrote {len(rows)} rows -> {out_path}")
    return out_path


if __name__ == "__main__":
    run()

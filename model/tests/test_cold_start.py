"""
Cold-start generalization tests.

The default 80/10/10 random split lets the model see most drugs and most
genes in training. The interesting question is: does the model generalize
to drugs (or genes) it has *never* seen interact during training?

We bucket test pairs into:
  * SEEN_BOTH        — both drug and gene appear in training positives
  * UNSEEN_DRUG      — drug never appears in any training positive (cold drug)
  * UNSEEN_GENE      — gene never appears in any training positive (cold gene)
  * UNSEEN_BOTH      — neither appears in training positives

For each bucket we report AUROC, AP (macro), macro-F1, accuracy, support.

Note: we use the existing splits (we do not re-split). So buckets are
identified after-the-fact based on training-edge participation.

Output:
  results/cold_start.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from test_utils import (
    NUM_CLASSES,
    NO_INTERACTION,
    RESULTS_DIR,
    get_split_predictions,
    load_or_train,
)


def _participation(splits):
    """Drugs / genes that participate in at least one TRAINING positive edge."""
    pairs, labels = splits["train"]
    pairs = pairs.numpy()
    labels = labels.numpy()
    pos_mask = labels != NO_INTERACTION
    train_drugs = set(pairs[pos_mask, 0].tolist())
    train_genes = set(pairs[pos_mask, 1].tolist())
    return train_drugs, train_genes


def _bucket_metrics(probs, labels, idx, name):
    if len(idx) == 0:
        return {"bucket": name, "n": 0,
                "auroc": "", "ap": "", "macro_f1": "", "accuracy": ""}
    p = probs[idx]
    y = labels[idx]
    pred = p.argmax(axis=1)
    acc = float((pred == y).mean())
    f1m = float(f1_score(y, pred, average="macro", zero_division=0,
                         labels=list(range(NUM_CLASSES))))
    try:
        au = roc_auc_score(y, p, multi_class="ovr", average="macro",
                           labels=list(range(NUM_CLASSES)))
    except ValueError:
        au = float("nan")
    try:
        y_oh = np.eye(NUM_CLASSES)[y]
        ap = average_precision_score(y_oh, p, average="macro")
    except ValueError:
        ap = float("nan")
    return {
        "bucket": name,
        "n": int(len(idx)),
        "auroc": "" if np.isnan(au) else round(au, 6),
        "ap": "" if np.isnan(ap) else round(ap, 6),
        "macro_f1": round(f1m, 6),
        "accuracy": round(acc, 6),
    }


def run(out_path: Path | None = None):
    out_path = out_path or (RESULTS_DIR / "cold_start.csv")
    model, graph, splits, info, _ = load_or_train()
    train_drugs, train_genes = _participation(splits)

    rows = []
    for split_name in ("val", "test"):
        pred = get_split_predictions(model, graph, splits, split_name)
        pairs = pred["pairs"]
        labels = pred["labels"]
        probs = pred["probs"]

        drug_seen = np.array([d in train_drugs for d in pairs[:, 0]])
        gene_seen = np.array([g in train_genes for g in pairs[:, 1]])

        buckets = {
            "SEEN_BOTH":   np.where(drug_seen & gene_seen)[0],
            "UNSEEN_DRUG": np.where(~drug_seen & gene_seen)[0],
            "UNSEEN_GENE": np.where(drug_seen & ~gene_seen)[0],
            "UNSEEN_BOTH": np.where(~drug_seen & ~gene_seen)[0],
        }
        for name, idx in buckets.items():
            r = _bucket_metrics(probs, labels, idx, name)
            r["split"] = split_name
            rows.append(r)

    fieldnames = ["split", "bucket", "n", "auroc", "ap", "macro_f1", "accuracy"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"[cold_start] wrote {len(rows)} rows -> {out_path}")
    return out_path


if __name__ == "__main__":
    run()

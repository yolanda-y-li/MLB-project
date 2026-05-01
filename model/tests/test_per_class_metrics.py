"""
Per-class evaluation.

Reports precision, recall, F1, support, AUROC (one-vs-rest), and AP
for each of the four classes individually. Also reports macro and weighted
averages so you can compare with the headline numbers.

Output:
  results/per_class_metrics.csv  — one row per class + macro/weighted rows
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from test_utils import (
    CLASS_NAMES,
    NUM_CLASSES,
    RESULTS_DIR,
    get_split_predictions,
    load_or_train,
)


def _safe_auc(y_true_bin, y_score):
    if y_true_bin.sum() == 0 or y_true_bin.sum() == len(y_true_bin):
        return float("nan")
    try:
        return roc_auc_score(y_true_bin, y_score)
    except ValueError:
        return float("nan")


def _safe_ap(y_true_bin, y_score):
    if y_true_bin.sum() == 0:
        return float("nan")
    try:
        return average_precision_score(y_true_bin, y_score)
    except ValueError:
        return float("nan")


def run(out_path: Path | None = None):
    out_path = out_path or (RESULTS_DIR / "per_class_metrics.csv")
    model, graph, splits, info, _ = load_or_train()

    rows = []
    for split_name in ("train", "val", "test"):
        pred = get_split_predictions(model, graph, splits, split_name)
        y_true = pred["labels"]
        y_pred = pred["preds"]
        probs = pred["probs"]

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(NUM_CLASSES)), zero_division=0
        )

        for c in range(NUM_CLASSES):
            y_bin = (y_true == c).astype(int)
            score = probs[:, c]
            rows.append({
                "split": split_name,
                "class_id": c,
                "class_name": CLASS_NAMES[c],
                "precision": round(float(precision[c]), 6),
                "recall": round(float(recall[c]), 6),
                "f1": round(float(f1[c]), 6),
                "support": int(support[c]),
                "auroc_ovr": round(_safe_auc(y_bin, score), 6) if not np.isnan(_safe_auc(y_bin, score)) else "",
                "ap": round(_safe_ap(y_bin, score), 6) if not np.isnan(_safe_ap(y_bin, score)) else "",
            })

        # Macro and weighted averages
        for avg in ("macro", "weighted"):
            p, r, f, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=list(range(NUM_CLASSES)),
                average=avg, zero_division=0,
            )
            try:
                au = roc_auc_score(y_true, probs, multi_class="ovr", average=avg)
            except ValueError:
                au = float("nan")
            try:
                y_oh = np.eye(NUM_CLASSES)[y_true]
                ap = average_precision_score(y_oh, probs, average=avg)
            except ValueError:
                ap = float("nan")
            rows.append({
                "split": split_name,
                "class_id": -1,
                "class_name": f"{avg}_avg",
                "precision": round(float(p), 6),
                "recall": round(float(r), 6),
                "f1": round(float(f), 6),
                "support": int(len(y_true)),
                "auroc_ovr": round(au, 6) if not np.isnan(au) else "",
                "ap": round(ap, 6) if not np.isnan(ap) else "",
            })

    fieldnames = ["split", "class_id", "class_name", "precision", "recall",
                  "f1", "support", "auroc_ovr", "ap"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"[per_class_metrics] wrote {len(rows)} rows -> {out_path}")
    return out_path


if __name__ == "__main__":
    run()

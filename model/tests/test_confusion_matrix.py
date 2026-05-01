"""
Confusion matrix tests.

Writes both raw counts and row-normalized confusion matrices for each split
into a single tidy CSV.

Output:
  results/confusion_matrix.csv
    columns: split, normalized, true_class, pred_class, count
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

from test_utils import (
    CLASS_NAMES,
    NUM_CLASSES,
    RESULTS_DIR,
    get_split_predictions,
    load_or_train,
)


def run(out_path: Path | None = None):
    out_path = out_path or (RESULTS_DIR / "confusion_matrix.csv")
    model, graph, splits, info, _ = load_or_train()

    rows = []
    for split_name in ("train", "val", "test"):
        pred = get_split_predictions(model, graph, splits, split_name)
        y_true = pred["labels"]
        y_pred = pred["preds"]

        cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
        cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                rows.append({
                    "split": split_name,
                    "normalized": 0,
                    "true_class": CLASS_NAMES[i],
                    "pred_class": CLASS_NAMES[j],
                    "count": int(cm[i, j]),
                })
                rows.append({
                    "split": split_name,
                    "normalized": 1,
                    "true_class": CLASS_NAMES[i],
                    "pred_class": CLASS_NAMES[j],
                    "count": round(float(cm_norm[i, j]), 6),
                })

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "normalized", "true_class",
                                          "pred_class", "count"])
        w.writeheader()
        w.writerows(rows)

    print(f"[confusion_matrix] wrote {len(rows)} rows -> {out_path}")
    return out_path


if __name__ == "__main__":
    run()

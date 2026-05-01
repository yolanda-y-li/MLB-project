"""
Threshold sensitivity for the binary "any interaction vs no_interaction" task.

The default classifier picks argmax over 4 classes. But for drug-discovery
screening you really care about the binary call: is there ANY interaction
between this (drug, gene) pair? We sweep the threshold on
P(interaction) = 1 - P(no_interaction) and report:

  * precision, recall, F1 at each threshold
  * accuracy
  * positives predicted at each threshold
  * the optimal F1 threshold

Output:
  results/threshold_sensitivity.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from test_utils import (
    NO_INTERACTION,
    RESULTS_DIR,
    get_split_predictions,
    load_or_train,
)


def run(out_path: Path | None = None, thresholds: np.ndarray | None = None):
    out_path = out_path or (RESULTS_DIR / "threshold_sensitivity.csv")
    if thresholds is None:
        thresholds = np.round(np.linspace(0.05, 0.95, 19), 3)
    model, graph, splits, info, _ = load_or_train()

    rows = []
    for split_name in ("train", "val", "test"):
        pred = get_split_predictions(model, graph, splits, split_name)
        labels = pred["labels"]
        probs = pred["probs"]
        # binary y: 1 = interaction (any class != NO_INTERACTION)
        y_true = (labels != NO_INTERACTION).astype(int)
        # P(interaction) = 1 - P(no_interaction)
        p_int = 1.0 - probs[:, NO_INTERACTION]

        for thr in thresholds:
            y_pred = (p_int >= thr).astype(int)
            p, r, f, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )
            acc = float((y_pred == y_true).mean())
            rows.append({
                "split": split_name,
                "threshold": float(thr),
                "precision": round(float(p), 6),
                "recall": round(float(r), 6),
                "f1": round(float(f), 6),
                "accuracy": round(acc, 6),
                "n_predicted_positive": int(y_pred.sum()),
                "n_actual_positive": int(y_true.sum()),
                "n_total": int(len(y_true)),
            })

    # Identify best F1 threshold per split
    best_per_split = {}
    for r in rows:
        s = r["split"]
        if s not in best_per_split or r["f1"] > best_per_split[s]["f1"]:
            best_per_split[s] = r
    for s, r in best_per_split.items():
        rows.append({**r, "threshold": f"BEST_F1@{r['threshold']:.3f}"})

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"[threshold_sensitivity] wrote {len(rows)} rows -> {out_path}")
    return out_path


if __name__ == "__main__":
    run()

"""
Per-edge-type performance.

The three positive interaction classes are very biologically different:
  * binds          (typically physical binding — Hetionet's biggest class)
  * upregulates    (drug increases gene expression)
  * downregulates  (drug decreases gene expression)

We compute, for each true edge type:
  * 1-vs-rest accuracy on test (correctly classified into its own class)
  * top-1 confusion (most common wrong prediction)
  * mean predicted probability on the true class
  * AUROC (one-vs-rest) restricted to "this class vs no-interaction"

Output:
  results/per_edge_type.csv
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from test_utils import (
    CLASS_NAMES,
    INTERACTION_LABELS,
    NO_INTERACTION,
    RESULTS_DIR,
    get_split_predictions,
    load_or_train,
)


def run(out_path: Path | None = None):
    out_path = out_path or (RESULTS_DIR / "per_edge_type.csv")
    model, graph, splits, info, _ = load_or_train()

    rows = []
    for split_name in ("val", "test"):
        pred = get_split_predictions(model, graph, splits, split_name)
        labels = pred["labels"]
        probs = pred["probs"]
        preds = pred["preds"]

        for edge_label in INTERACTION_LABELS:
            mask = labels == edge_label
            n = int(mask.sum())
            if n == 0:
                rows.append({
                    "split": split_name,
                    "true_class": CLASS_NAMES[edge_label],
                    "n": 0,
                    "accuracy_correct_class": "",
                    "mean_prob_true_class": "",
                    "top1_confusion_class": "",
                    "top1_confusion_share": "",
                    "auroc_vs_no_interaction": "",
                })
                continue
            correct = float((preds[mask] == edge_label).mean())
            mean_p = float(probs[mask, edge_label].mean())

            # Top-1 confusion
            wrongs = preds[mask][preds[mask] != edge_label]
            if len(wrongs):
                top, share = Counter(wrongs.tolist()).most_common(1)[0]
                top1_class = CLASS_NAMES[top]
                top1_share = round(share / n, 6)
            else:
                top1_class = ""
                top1_share = ""

            # AUROC: this class vs NO_INTERACTION  (restricted)
            sub_mask = (labels == edge_label) | (labels == NO_INTERACTION)
            yb = (labels[sub_mask] == edge_label).astype(int)
            ps = probs[sub_mask, edge_label]
            try:
                au = roc_auc_score(yb, ps)
            except ValueError:
                au = float("nan")

            rows.append({
                "split": split_name,
                "true_class": CLASS_NAMES[edge_label],
                "n": n,
                "accuracy_correct_class": round(correct, 6),
                "mean_prob_true_class": round(mean_p, 6),
                "top1_confusion_class": top1_class,
                "top1_confusion_share": top1_share,
                "auroc_vs_no_interaction": "" if np.isnan(au) else round(au, 6),
            })

    fieldnames = ["split", "true_class", "n", "accuracy_correct_class",
                  "mean_prob_true_class", "top1_confusion_class",
                  "top1_confusion_share", "auroc_vs_no_interaction"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"[per_edge_type] wrote {len(rows)} rows -> {out_path}")
    return out_path


if __name__ == "__main__":
    run()

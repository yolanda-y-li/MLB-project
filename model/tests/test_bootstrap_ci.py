"""
Bootstrap confidence intervals for the headline metrics.

Single-number metrics like AUROC/AP/F1 hide a lot of variance. A bootstrap
gives us 95% CIs by resampling the test set with replacement and recomputing
each metric. We report mean, std, 2.5% / 97.5% percentiles for AUROC, AP,
macro-F1, and binary AUROC for the (any-interaction vs no-interaction) task.

Output:
  results/bootstrap_ci.csv
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


def _metrics_on_indices(probs, labels, idx):
    p = probs[idx]
    y = labels[idx]
    pred = p.argmax(axis=1)
    out = {}
    try:
        out["auroc_macro"] = roc_auc_score(y, p, multi_class="ovr", average="macro",
                                           labels=list(range(NUM_CLASSES)))
    except ValueError:
        out["auroc_macro"] = float("nan")
    try:
        out["ap_macro"] = average_precision_score(np.eye(NUM_CLASSES)[y], p, average="macro")
    except ValueError:
        out["ap_macro"] = float("nan")
    out["f1_macro"] = f1_score(y, pred, average="macro", zero_division=0,
                               labels=list(range(NUM_CLASSES)))
    # Binary "any interaction" task
    yb = (y != NO_INTERACTION).astype(int)
    pb = 1.0 - p[:, NO_INTERACTION]
    try:
        out["auroc_binary"] = roc_auc_score(yb, pb)
    except ValueError:
        out["auroc_binary"] = float("nan")
    try:
        out["ap_binary"] = average_precision_score(yb, pb)
    except ValueError:
        out["ap_binary"] = float("nan")
    return out


def run(out_path: Path | None = None, n_boot: int = 1000, seed: int = 0):
    out_path = out_path or (RESULTS_DIR / "bootstrap_ci.csv")
    model, graph, splits, info, _ = load_or_train()
    rng = np.random.default_rng(seed)

    rows = []
    for split_name in ("val", "test"):
        pred = get_split_predictions(model, graph, splits, split_name)
        labels = pred["labels"]
        probs = pred["probs"]
        n = len(labels)

        # Point estimate
        point = _metrics_on_indices(probs, labels, np.arange(n))

        # Bootstrap
        boot = {k: [] for k in point.keys()}
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            res = _metrics_on_indices(probs, labels, idx)
            for k, v in res.items():
                if not np.isnan(v):
                    boot[k].append(v)

        for k, vals in boot.items():
            arr = np.array(vals) if vals else np.array([np.nan])
            rows.append({
                "split": split_name,
                "metric": k,
                "point_estimate": "" if np.isnan(point[k]) else round(point[k], 6),
                "boot_mean": "" if len(vals) == 0 else round(float(arr.mean()), 6),
                "boot_std": "" if len(vals) == 0 else round(float(arr.std(ddof=1)) if len(vals) > 1 else 0.0, 6),
                "ci_lower_2.5": "" if len(vals) == 0 else round(float(np.quantile(arr, 0.025)), 6),
                "ci_upper_97.5": "" if len(vals) == 0 else round(float(np.quantile(arr, 0.975)), 6),
                "n_boot_samples": len(vals),
                "n_test": n,
            })

    fieldnames = ["split", "metric", "point_estimate", "boot_mean", "boot_std",
                  "ci_lower_2.5", "ci_upper_97.5", "n_boot_samples", "n_test"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"[bootstrap_ci] wrote {len(rows)} rows -> {out_path}")
    return out_path


if __name__ == "__main__":
    run()

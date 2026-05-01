"""
Calibration tests.

For a softmax classifier, "well-calibrated" means: among predictions where
the model says it is 80% confident, ~80% are actually correct. We measure:

  * ECE  — Expected Calibration Error (max-prob bins)
  * MCE  — Maximum Calibration Error
  * Brier score (multi-class generalisation: mean squared error of probs vs. one-hot)
  * NLL  — negative log-likelihood
  * Reliability bins — confidence bin → empirical accuracy + count

Outputs:
  results/calibration_summary.csv  — one row per split with ECE/MCE/Brier/NLL
  results/calibration_bins.csv     — reliability bins per split
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from test_utils import (
    NUM_CLASSES,
    RESULTS_DIR,
    get_split_predictions,
    load_or_train,
)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    """Return (ECE, MCE, list of (bin_lo, bin_hi, mean_conf, mean_acc, count))."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(probs)
    ece = 0.0
    mce = 0.0
    bin_rows = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences <= hi)
        count = int(in_bin.sum())
        if count == 0:
            bin_rows.append((lo, hi, float("nan"), float("nan"), 0))
            continue
        mean_conf = float(confidences[in_bin].mean())
        mean_acc = float(accuracies[in_bin].mean())
        gap = abs(mean_conf - mean_acc)
        ece += gap * count / n
        mce = max(mce, gap)
        bin_rows.append((lo, hi, mean_conf, mean_acc, count))
    return ece, mce, bin_rows


def brier_multiclass(probs: np.ndarray, labels: np.ndarray) -> float:
    onehot = np.eye(NUM_CLASSES)[labels]
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(probs[np.arange(len(labels)), labels], eps, 1.0)
    return float(-np.log(p).mean())


def run(summary_path: Path | None = None, bins_path: Path | None = None,
        n_bins: int = 15):
    summary_path = summary_path or (RESULTS_DIR / "calibration_summary.csv")
    bins_path = bins_path or (RESULTS_DIR / "calibration_bins.csv")
    model, graph, splits, info, _ = load_or_train()

    summary_rows = []
    bin_rows_all = []
    for split_name in ("train", "val", "test"):
        pred = get_split_predictions(model, graph, splits, split_name)
        probs = pred["probs"]
        labels = pred["labels"]
        ece, mce, bin_rows = expected_calibration_error(probs, labels, n_bins=n_bins)
        summary_rows.append({
            "split": split_name,
            "ece": round(ece, 6),
            "mce": round(mce, 6),
            "brier": round(brier_multiclass(probs, labels), 6),
            "nll": round(nll(probs, labels), 6),
            "n": int(len(labels)),
            "n_bins": n_bins,
        })
        for lo, hi, mean_conf, mean_acc, count in bin_rows:
            bin_rows_all.append({
                "split": split_name,
                "bin_lo": round(lo, 6),
                "bin_hi": round(hi, 6),
                "mean_confidence": "" if np.isnan(mean_conf) else round(mean_conf, 6),
                "mean_accuracy": "" if np.isnan(mean_acc) else round(mean_acc, 6),
                "count": count,
            })

    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "ece", "mce", "brier", "nll", "n", "n_bins"])
        w.writeheader()
        w.writerows(summary_rows)

    with open(bins_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "bin_lo", "bin_hi",
                                          "mean_confidence", "mean_accuracy", "count"])
        w.writeheader()
        w.writerows(bin_rows_all)

    print(f"[calibration] summary -> {summary_path}")
    print(f"[calibration] bins    -> {bins_path}")


if __name__ == "__main__":
    run()

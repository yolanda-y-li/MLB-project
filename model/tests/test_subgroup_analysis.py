"""
Subgroup analysis by node degree.

Drugs and genes vary widely in how many edges they have in the training
graph. Models often do well on hub nodes (lots of training signal) and
poorly on rare nodes. We bucket test pairs by drug-degree and gene-degree
quantiles (Q1..Q4) and report metrics per bucket.

Output:
  results/subgroup_by_drug_degree.csv
  results/subgroup_by_gene_degree.csv
"""

from __future__ import annotations

import csv
from collections import Counter
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


def _train_degrees(splits, n_drugs, n_genes):
    pairs, labels = splits["train"]
    pairs = pairs.numpy()
    labels = labels.numpy()
    pos = labels != NO_INTERACTION
    drug_deg = Counter(pairs[pos, 0].tolist())
    gene_deg = Counter(pairs[pos, 1].tolist())
    drug_arr = np.array([drug_deg.get(i, 0) for i in range(n_drugs)])
    gene_arr = np.array([gene_deg.get(i, 0) for i in range(n_genes)])
    return drug_arr, gene_arr


def _quantile_buckets(values: np.ndarray, n_q: int = 4):
    """Return labels Q1..Qn for each value (Q1 = lowest)."""
    if len(values) == 0:
        return np.array([], dtype=int)
    qs = np.quantile(values, np.linspace(0, 1, n_q + 1))
    qs[0] = -np.inf
    qs[-1] = np.inf
    bucket = np.digitize(values, qs[1:-1], right=False) + 1
    return np.clip(bucket, 1, n_q)


def _metrics(probs, labels):
    if len(labels) == 0:
        return {"auroc_macro": "", "ap_macro": "", "f1_macro": "", "accuracy": ""}
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
    return {k: ("" if isinstance(v, float) and np.isnan(v) else round(v, 6) if isinstance(v, float) else v)
            for k, v in out.items()}


def run(by_drug_path: Path | None = None, by_gene_path: Path | None = None):
    by_drug_path = by_drug_path or (RESULTS_DIR / "subgroup_by_drug_degree.csv")
    by_gene_path = by_gene_path or (RESULTS_DIR / "subgroup_by_gene_degree.csv")
    model, graph, splits, info, _ = load_or_train()

    drug_deg, gene_deg = _train_degrees(splits, info["n_drugs"], info["n_genes"])
    pred = get_split_predictions(model, graph, splits, "test")
    pairs = pred["pairs"]
    labels = pred["labels"]
    probs = pred["probs"]

    drug_b = _quantile_buckets(drug_deg[pairs[:, 0]])
    gene_b = _quantile_buckets(gene_deg[pairs[:, 1]])

    fieldnames = ["bucket", "n", "mean_train_degree", "auroc_macro",
                  "ap_macro", "f1_macro", "accuracy"]

    def _write(buckets, deg_arr, pair_col, path):
        rows = []
        for q in range(1, 5):
            idx = np.where(buckets == q)[0]
            n = len(idx)
            if n == 0:
                rows.append({"bucket": f"Q{q}", "n": 0, "mean_train_degree": "",
                             "auroc_macro": "", "ap_macro": "", "f1_macro": "",
                             "accuracy": ""})
                continue
            md = float(deg_arr[pairs[idx, pair_col]].mean())
            m = _metrics(probs[idx], labels[idx])
            rows.append({"bucket": f"Q{q}", "n": n,
                         "mean_train_degree": round(md, 3), **m})
        # Overall
        m = _metrics(probs, labels)
        rows.append({"bucket": "ALL", "n": int(len(labels)),
                     "mean_train_degree": round(float(deg_arr[pairs[:, pair_col]].mean()), 3),
                     **m})
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    _write(drug_b, drug_deg, 0, by_drug_path)
    _write(gene_b, gene_deg, 1, by_gene_path)

    print(f"[subgroup] drug-degree -> {by_drug_path}")
    print(f"[subgroup] gene-degree -> {by_gene_path}")


if __name__ == "__main__":
    run()

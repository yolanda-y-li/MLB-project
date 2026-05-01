"""
Ranking metrics — Hits@K and MRR.

For each gene that has at least one positive test interaction, score that
gene against ALL drugs in the catalog using P(interaction) = 1 - P(no).
Rank the drugs and check whether the true positive drug(s) appear in the top K.
This mirrors the practical use case "given a gene, which drugs should we
investigate first?" — far more diagnostic than a flat F1 score.

We compute, for K in {1, 5, 10, 20, 50, 100}:
  * Hits@K  — fraction of positive (gene, drug) test pairs whose drug appears
              in the top-K predictions for that gene
  * MRR     — mean reciprocal rank of the true drug
  * Mean rank, median rank

Drugs that are positives for the gene in the train/val splits are excluded
from the candidate set (filtered ranking) to avoid them displacing the test
positive in the ranking.

Output:
  results/ranking_metrics.csv
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from test_utils import (
    NO_INTERACTION,
    RESULTS_DIR,
    load_or_train,
)


def _all_positive_lookup(splits) -> dict[int, set[int]]:
    """gene_idx -> set of drug_idx that interact with it (any class), across all splits."""
    pos = defaultdict(set)
    for split_name in ("train", "val", "test"):
        pairs, labels = splits[split_name]
        pairs = pairs.numpy()
        labels = labels.numpy()
        for (drug, gene), lbl in zip(pairs, labels):
            if lbl != NO_INTERACTION:
                pos[int(gene)].add(int(drug))
    return pos


def _test_positives_per_gene(splits) -> dict[int, set[int]]:
    pos = defaultdict(set)
    pairs, labels = splits["test"]
    for (drug, gene), lbl in zip(pairs.numpy(), labels.numpy()):
        if lbl != NO_INTERACTION:
            pos[int(gene)].add(int(drug))
    return pos


@torch.no_grad()
def _score_gene_against_all_drugs(model, x_dict, gene_idx: int, n_drugs: int,
                                   device, batch_size: int = 4096) -> np.ndarray:
    """Returns P(interaction) for this gene against every drug, length n_drugs."""
    gene_emb = x_dict["gene"][gene_idx]  # [H]
    out = np.empty(n_drugs, dtype=np.float32)
    for start in range(0, n_drugs, batch_size):
        end = min(start + batch_size, n_drugs)
        drug_idx = torch.arange(start, end, device=device)
        de = x_dict["compound"][drug_idx]                          # [B, H]
        ge = gene_emb.unsqueeze(0).expand(end - start, -1)         # [B, H]
        logits = model.classifier(torch.cat([de, ge], dim=-1))     # [B, 4]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        out[start:end] = 1.0 - probs[:, NO_INTERACTION]
    return out


def run(out_path: Path | None = None, k_list=(1, 5, 10, 20, 50, 100),
        max_genes: int | None = None):
    out_path = out_path or (RESULTS_DIR / "ranking_metrics.csv")
    model, graph, splits, info, _ = load_or_train()
    device = next(model.parameters()).device

    n_drugs = info["n_drugs"]
    all_pos = _all_positive_lookup(splits)
    test_pos = _test_positives_per_gene(splits)

    genes_with_test = sorted(test_pos.keys())
    if max_genes is not None:
        genes_with_test = genes_with_test[:max_genes]
    print(f"[ranking_metrics] evaluating {len(genes_with_test)} genes with test positives")

    # Pre-encode once
    model.eval()
    x_dict = model.encode(graph.edge_index_dict)

    ranks = []  # per (gene, true_drug) pair
    for gi, gene_idx in enumerate(genes_with_test):
        scores = _score_gene_against_all_drugs(model, x_dict, gene_idx, n_drugs, device)
        # Filtered ranking: mask out positives that are NOT in test set for this gene
        train_val_positives = all_pos[gene_idx] - test_pos[gene_idx]
        mask = np.ones(n_drugs, dtype=bool)
        for d in train_val_positives:
            mask[d] = False
        # For each true test drug, compute rank among remaining candidates
        for true_drug in test_pos[gene_idx]:
            # rank: 1-indexed position when sorted desc by score
            valid_mask = mask.copy()
            valid_mask[true_drug] = True  # ensure true drug is included
            valid_scores = scores[valid_mask]
            valid_indices = np.where(valid_mask)[0]
            # higher score = better. rank is 1 + count of strictly higher scores
            true_score = scores[true_drug]
            rank = 1 + int((valid_scores > true_score).sum())
            ranks.append(rank)
        if (gi + 1) % 200 == 0:
            print(f"  ... {gi + 1}/{len(genes_with_test)} genes done")

    ranks_arr = np.array(ranks, dtype=np.int64)
    rows = []
    if len(ranks_arr) == 0:
        print("[ranking_metrics] WARNING: no test positives found")
    else:
        for k in k_list:
            hits = float((ranks_arr <= k).mean())
            rows.append({
                "metric": f"hits@{k}",
                "value": round(hits, 6),
                "n_queries": len(ranks_arr),
            })
        rows.append({
            "metric": "mrr",
            "value": round(float((1.0 / ranks_arr).mean()), 6),
            "n_queries": len(ranks_arr),
        })
        rows.append({
            "metric": "mean_rank",
            "value": round(float(ranks_arr.mean()), 4),
            "n_queries": len(ranks_arr),
        })
        rows.append({
            "metric": "median_rank",
            "value": float(np.median(ranks_arr)),
            "n_queries": len(ranks_arr),
        })

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "value", "n_queries"])
        w.writeheader()
        w.writerows(rows)

    print(f"[ranking_metrics] wrote {len(rows)} rows -> {out_path}")
    return out_path


if __name__ == "__main__":
    run()

"""
Data loading and graph construction for gene-drug interaction prediction.

Graph nodes: gene, compound, pharmacologic_class, gene_family
Graph edges:
  compound  -[binds/upregulates/downregulates]-> gene      (from Hetionet)
  pharmacologic_class -[includes]-> compound               (from Hetionet)
  gene -[belongs_to]-> gene_family                         (from HGNC)
  gene_family -[child_of]-> gene_family  (hierarchy)       (from HGNC)

Reverse edges are added for each relation so information flows both ways.
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

DATA_DIR = Path(__file__).parent.parent / "Data"

# Hetionet edge-type codes → relation names
EDGE_CODE_TO_REL = {
    "CbG": "binds",
    "CuG": "upregulates",
    "CdG": "downregulates",
}

# Interaction class labels (0-2 positive, 3 = no interaction)
LABEL_MAP = {"binds": 0, "upregulates": 1, "downregulates": 2}
NO_INTERACTION = 3
NUM_CLASSES = 4


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_data(neg_strategy: str = "edge_swap", neg_ratio: float = 1.0, seed: int = 42):
    """
    Load all data files, construct the heterogeneous graph, and produce
    train / val / test pair tensors with labels.

    Args:
        neg_strategy: "edge_swap" (recommended) or "random"
        neg_ratio:    number of negatives per positive example
        seed:         random seed

    Returns:
        train_graph:  HeteroData – message-passing graph built from training
                      gene-drug edges + all auxiliary edges
        splits:       dict with keys "train", "val", "test", each a tuple
                      (pairs [N,2] long, labels [N] long)
                      where pairs[:, 0] = drug_idx, pairs[:, 1] = gene_idx
        info:         dict with node counts and id2idx mappings
    """
    random.seed(seed)
    np.random.seed(seed)

    # ------------------------------------------------------------------
    # Load node tables
    # ------------------------------------------------------------------
    gene_df = pd.read_csv(
        DATA_DIR / "nodes/gene_nodes.tsv", sep="\t", header=None,
        names=["id", "symbol", "type"],
    )
    drug_df = pd.read_csv(
        DATA_DIR / "nodes/drug_nodes.tsv", sep="\t", header=None,
        names=["id", "name", "type"],
    )
    class_df = pd.read_csv(
        DATA_DIR / "nodes/drug_class_nodes.tsv", sep="\t", header=None,
        names=["id", "name", "type"],
    )

    gene_id2idx  = {gid: i for i, gid in enumerate(gene_df["id"])}
    drug_id2idx  = {did: i for i, did in enumerate(drug_df["id"])}
    class_id2idx = {cid: i for i, cid in enumerate(class_df["id"])}

    n_genes  = len(gene_df)
    n_drugs  = len(drug_df)
    n_classes = len(class_df)

    # ------------------------------------------------------------------
    # Load gene-drug edges
    # ------------------------------------------------------------------
    gd_raw = pd.read_csv(
        DATA_DIR / "edges/gene_drug_edges.tsv", sep="\t", header=None,
        names=["compound", "edge_code", "gene"],
    )
    gd_raw["rel"]      = gd_raw["edge_code"].map(EDGE_CODE_TO_REL)
    gd_raw["label"]    = gd_raw["rel"].map(LABEL_MAP)
    gd_raw["drug_idx"] = gd_raw["compound"].map(drug_id2idx)
    gd_raw["gene_idx"] = gd_raw["gene"].map(gene_id2idx)
    gd_raw = gd_raw.dropna(subset=["drug_idx", "gene_idx", "label"]).copy()
    gd_raw = gd_raw.astype({"drug_idx": int, "gene_idx": int, "label": int})

    # ------------------------------------------------------------------
    # Train / val / test split of positive edges  (80 / 10 / 10)
    # ------------------------------------------------------------------
    gd_raw = gd_raw.sample(frac=1, random_state=seed).reset_index(drop=True)
    n_total = len(gd_raw)
    n_train = int(0.8 * n_total)
    n_val   = int(0.1 * n_total)

    train_pos = gd_raw.iloc[:n_train].copy()
    val_pos   = gd_raw.iloc[n_train : n_train + n_val].copy()
    test_pos  = gd_raw.iloc[n_train + n_val :].copy()

    # Set of ALL known (drug_idx, gene_idx) positive pairs – used to
    # ensure negatives don't accidentally include known interactions
    all_positive_set = set(zip(gd_raw["drug_idx"], gd_raw["gene_idx"]))

    # ------------------------------------------------------------------
    # Build heterogeneous graph (message-passing uses training edges only
    # for gene-drug relations to avoid data leakage)
    # ------------------------------------------------------------------
    train_graph = _build_graph(
        train_pos, drug_id2idx, class_id2idx, gene_id2idx,
        n_genes, n_drugs, n_classes,
    )

    # ------------------------------------------------------------------
    # Negative sampling + assemble split tensors
    # ------------------------------------------------------------------
    splits = {}
    for name, pos_df in [("train", train_pos), ("val", val_pos), ("test", test_pos)]:
        n_neg = max(1, int(len(pos_df) * neg_ratio))
        if neg_strategy == "edge_swap":
            neg_pairs = _edge_swap_negatives(pos_df, n_neg, all_positive_set, n_drugs, n_genes)
        else:
            neg_pairs = _random_negatives(n_neg, all_positive_set, n_drugs, n_genes)

        pos_pairs  = list(zip(pos_df["drug_idx"], pos_df["gene_idx"]))
        pos_labels = list(pos_df["label"])
        all_pairs  = pos_pairs  + neg_pairs
        all_labels = pos_labels + [NO_INTERACTION] * len(neg_pairs)

        splits[name] = (
            torch.tensor(all_pairs,  dtype=torch.long),
            torch.tensor(all_labels, dtype=torch.long),
        )

    info = {
        "n_genes":    n_genes,
        "n_drugs":    n_drugs,
        "n_classes":  n_classes,
        "n_families": train_graph["gene_family"].num_nodes,
        "gene_id2idx": gene_id2idx,
        "drug_id2idx": drug_id2idx,
    }
    return train_graph, splits, info


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_graph(train_pos_df, drug_id2idx, class_id2idx, gene_id2idx,
                 n_genes, n_drugs, n_classes):
    """Construct HeteroData used for R-GCN message passing."""
    data = HeteroData()
    data["gene"].num_nodes              = n_genes
    data["compound"].num_nodes          = n_drugs
    data["pharmacologic_class"].num_nodes = n_classes

    # --- Gene-drug edges (training edges only) ---
    for code, rel in EDGE_CODE_TO_REL.items():
        sub = train_pos_df[train_pos_df["edge_code"] == code]
        if len(sub) == 0:
            continue
        src = torch.tensor(sub["drug_idx"].values, dtype=torch.long)
        dst = torch.tensor(sub["gene_idx"].values,  dtype=torch.long)
        data["compound", rel, "gene"].edge_index             = torch.stack([src, dst])
        data["gene", f"rev_{rel}", "compound"].edge_index    = torch.stack([dst, src])

    # --- Pharmacologic class → compound edges ---
    cd_df = pd.read_csv(
        DATA_DIR / "edges/class_drug_edges.tsv", sep="\t", header=None,
        names=["pharm_class", "edge_code", "compound"],
    )
    cd_df["class_idx"] = cd_df["pharm_class"].map(class_id2idx)
    cd_df["drug_idx"]  = cd_df["compound"].map(drug_id2idx)
    cd_df = cd_df.dropna(subset=["class_idx", "drug_idx"]).astype(
        {"class_idx": int, "drug_idx": int}
    )
    src = torch.tensor(cd_df["class_idx"].values, dtype=torch.long)
    dst = torch.tensor(cd_df["drug_idx"].values,  dtype=torch.long)
    data["pharmacologic_class", "includes", "compound"].edge_index    = torch.stack([src, dst])
    data["compound", "rev_includes", "pharmacologic_class"].edge_index = torch.stack([dst, src])

    # --- Gene → gene_family edges (HGNC membership) ---
    gf_df = pd.read_csv(
        DATA_DIR / "edges/gene_family_edges.tsv", sep="\t",
    )
    # Column headers: "0" (gene id), "family_id", "parent_fam_id"
    gf_df = gf_df.rename(columns={"0": "gene_id"})
    gf_df = gf_df.dropna(subset=["gene_id", "family_id"]).copy()
    gf_df["gene_idx"]  = gf_df["gene_id"].map(gene_id2idx)
    gf_df = gf_df.dropna(subset=["gene_idx"]).copy()
    gf_df["family_id"] = gf_df["family_id"].astype(int)
    gf_df["gene_idx"]  = gf_df["gene_idx"].astype(int)

    # Build family_id → sequential index from all observed family IDs
    all_fam_ids = set(gf_df["family_id"].tolist())
    parent_ids  = set(
        gf_df["parent_fam_id"].dropna().astype(int).tolist()
    )
    all_fam_ids |= parent_ids
    fam_id2idx  = {fid: i for i, fid in enumerate(sorted(all_fam_ids))}
    n_families  = len(fam_id2idx)
    data["gene_family"].num_nodes = n_families

    gf_df["fam_idx"] = gf_df["family_id"].map(fam_id2idx)
    g_src = torch.tensor(gf_df["gene_idx"].values, dtype=torch.long)
    g_dst = torch.tensor(gf_df["fam_idx"].values,  dtype=torch.long)
    data["gene", "belongs_to", "gene_family"].edge_index         = torch.stack([g_src, g_dst])
    data["gene_family", "rev_belongs_to", "gene"].edge_index     = torch.stack([g_dst, g_src])

    # Family hierarchy edges (child → parent)
    hier = gf_df.dropna(subset=["parent_fam_id"]).copy()
    hier["parent_fam_id"]  = hier["parent_fam_id"].astype(int)
    hier["parent_fam_idx"] = hier["parent_fam_id"].map(fam_id2idx)
    hier = hier.dropna(subset=["parent_fam_idx"]).copy()
    hier["parent_fam_idx"] = hier["parent_fam_idx"].astype(int)

    hier_edges = hier[["fam_idx", "parent_fam_idx"]].drop_duplicates()
    h_src = torch.tensor(hier_edges["fam_idx"].values,        dtype=torch.long)
    h_dst = torch.tensor(hier_edges["parent_fam_idx"].values, dtype=torch.long)
    data["gene_family", "child_of", "gene_family"].edge_index = torch.stack([h_src, h_dst])

    return data


def _edge_swap_negatives(pos_df, n_neg, positive_set, n_drugs, n_genes, seed=None):
    """
    Generate hard negatives via edge swap.
    If (drug_a, gene_b) and (drug_c, gene_d) are positives, produce
    (drug_a, gene_d) and (drug_c, gene_b) as candidates if not in positive_set.
    Falls back to random sampling when edge swaps are exhausted.
    """
    pos_list = list(zip(pos_df["drug_idx"], pos_df["gene_idx"]))
    negatives: set = set()
    max_attempts = n_neg * 30

    for _ in range(max_attempts):
        if len(negatives) >= n_neg:
            break
        i, j = random.sample(range(len(pos_list)), 2)
        d_a, g_b = pos_list[i]
        d_c, g_d = pos_list[j]
        for cand in [(d_a, g_d), (d_c, g_b)]:
            if cand not in positive_set and cand not in negatives:
                negatives.add(cand)
                if len(negatives) >= n_neg:
                    break

    # Fall back to random to top up
    while len(negatives) < n_neg:
        cand = (random.randint(0, n_drugs - 1), random.randint(0, n_genes - 1))
        if cand not in positive_set and cand not in negatives:
            negatives.add(cand)

    return list(negatives)[:n_neg]


def _random_negatives(n_neg, positive_set, n_drugs, n_genes):
    """Generate negatives by random (drug, gene) pairing."""
    negatives: set = set()
    while len(negatives) < n_neg:
        cand = (random.randint(0, n_drugs - 1), random.randint(0, n_genes - 1))
        if cand not in positive_set and cand not in negatives:
            negatives.add(cand)
    return list(negatives)

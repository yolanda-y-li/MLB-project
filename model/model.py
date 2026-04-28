"""
R-GCN model for multi-class gene-drug interaction prediction.

Architecture:
  1. Learnable node embeddings for each node type (gene, compound,
     pharmacologic_class, gene_family) serve as initial node features.
  2. Two-layer Relational GCN encoder (HeteroConv + SAGEConv per relation)
     updates node representations via relation-specific message passing.
  3. Classification head: concat(drug_emb, gene_emb) → Linear → ReLU
     → Linear → logits over 4 classes (binds / upregulates /
     downregulates / no_interaction).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv

NUM_CLASSES = 4  # binds=0, upregulates=1, downregulates=2, no_interaction=3


class RGCNEncoder(nn.Module):
    """
    Heterogeneous R-GCN: one SAGEConv per relation type, wrapped in HeteroConv.
    SAGEConv with mean aggregation is equivalent to R-GCN's basis decomposition
    when each relation gets its own weight matrix.
    """

    def __init__(self, hidden_dim: int, num_layers: int, metadata: tuple,
                 dropout: float = 0.2):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {
                edge_type: SAGEConv((-1, -1), hidden_dim, aggr="mean")
                for edge_type in metadata[1]
            }
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

    def forward(self, x_dict: dict, edge_index_dict: dict) -> dict:
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            if i < self.num_layers - 1:
                x_dict = {k: F.relu(v)           for k, v in x_dict.items()}
                x_dict = {k: self.dropout(v)     for k, v in x_dict.items()}
        return x_dict


class GeneDrugRGCN(nn.Module):
    """
    Full model: learnable embeddings → R-GCN encoder → classification head.
    """

    def __init__(
        self,
        n_genes:    int,
        n_drugs:    int,
        n_classes:  int,
        n_families: int,
        hidden_dim: int   = 64,
        num_layers: int   = 2,
        dropout:    float = 0.2,
        metadata:   tuple = None,
    ):
        super().__init__()

        # Initial node features: one learnable embedding table per node type
        self.gene_emb    = nn.Embedding(n_genes,    hidden_dim)
        self.drug_emb    = nn.Embedding(n_drugs,    hidden_dim)
        self.class_emb   = nn.Embedding(n_classes,  hidden_dim)
        self.family_emb  = nn.Embedding(n_families, hidden_dim)

        self.encoder = RGCNEncoder(hidden_dim, num_layers, metadata, dropout)

        # Classifier: concat(drug, gene) embeddings → 4-class logits
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, NUM_CLASSES),
        )

        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self):
        for emb in [self.gene_emb, self.drug_emb, self.class_emb, self.family_emb]:
            nn.init.xavier_uniform_(emb.weight)

    def encode(self, edge_index_dict: dict) -> dict:
        """Run R-GCN over the full graph; returns updated x_dict."""
        x_dict = {
            "gene":               self.gene_emb.weight,
            "compound":           self.drug_emb.weight,
            "pharmacologic_class": self.class_emb.weight,
            "gene_family":        self.family_emb.weight,
        }
        return self.encoder(x_dict, edge_index_dict)

    def forward(self, edge_index_dict: dict, pairs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_index_dict: heterogeneous graph edge indices
            pairs:           [N, 2] long tensor – (drug_idx, gene_idx)

        Returns:
            logits: [N, 4]
        """
        x_dict   = self.encode(edge_index_dict)
        drug_emb = x_dict["compound"][pairs[:, 0]]
        gene_emb = x_dict["gene"][pairs[:, 1]]
        return self.classifier(torch.cat([drug_emb, gene_emb], dim=-1))

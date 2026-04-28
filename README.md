# MLB-Project

### Data

Below is the directory structure. You should only need to use data within the nodes and edges directories. There are different types of edges, so in case it was easier to have them separated, there is a directory that separates gene_drug_edges.tsv into three separate tsvs. 

If you want to run get_data.sh yourself, you need get_data.sh, join_gene_family.py, join_hgnc.py, and hetionet-v1.0-edges.sif.gz in the same directory and have a conda environment named ete3_env with requests and pandas installed. hetionet-v1.0-edges.sif.gz could not be curled and has to be downloaded directly from https://github.com/hetio/hetionet/blob/main/hetnet/tsv/hetionet-v1.0-edges.sif.gz. Also, made this specifically for my mac so I am sorry if it doesn't work with your architecture lol but you shouldn't need to rerun it anyway.

warning -  gene_family_edges.tsv may include gene duplicates bc some genes are in multiple families.

```
Data/
├── edges/
│   ├── class_drug_edges.tsv
│   ├── gene_family_edges.tsv
│   ├── gene_drug_edges.tsv
│   └── gene_drug_by_edge_type/
│       └── gene_drug_binds_edges.tsv
│       └── gene_drug_downreg_edges.tsv
│       └── gene_drug_upreg_edges.tsv
├── nodes/
│    ├── drug_class_nodes.tsv
│    ├── drug_nodes.tsv
│    ├── gene_family_nodes.tsv
│    └── gene_nodes.tsv
├── process_data/
│    ├── family.tsv
│    ├── gene_families.tsv
│    ├── gene_family_joined.tsv
│    ├── hgnc_genes.tsv
│    ├── hierarchy.tsv
│    └── gene_has_family.tsv
├── raw_data/
│    ├── family.csv
│    ├── hetionet-v1.0-edges.sif.gz
│    ├── hetionet-v1.0-nodes.tsv
│    ├── hgnc_complete_set.tsv
│    ├── hierarchy.csv
│    └── gene_has_family.csv
├── scripts/
│    ├── join_gene_family.py
│    └── join_hgnc.py
└── get_data.sh

```

---

### Model

Given a (gene, compound) pair, the model predicts whether the interaction is "binds", "upregulates", "downregulates", or "no interaction".

**Architecture:** Relational Graph Convolutional Network (R-GCN)

1. **Heterogeneous graph** — four node types (gene, compound, pharmacologic class, gene family) and eight relation types including the three gene-drug interaction types from Hetionet, pharmacologic class membership, HGNC gene family membership, and the gene family hierarchy.
2. **Learnable embeddings** — each node type gets its own embedding table (dim 64 by default) as initial node features, since no pre-computed features are used.
3. **R-GCN encoder** — two layers of `HeteroConv` with relation-specific `SAGEConv` weights. Each layer aggregates neighbor information separately per relation type, then sums across relations.
4. **Classification head** — the gene and compound embeddings are concatenated and passed through a two-layer MLP (Linear → ReLU → Linear) to produce logits over the four classes.
5. **Training** — cross-entropy loss, Adam optimizer, learning rate decay on plateau, early stopping on val AUROC. Negative examples are generated via **edge swap** (swapping genes between two known positives to create hard negatives).

**Results (default settings):** AUROC 0.857 · AP 0.648 · F1 0.631

#### Running the model

Install dependencies:
```bash
pip install torch torch-geometric scikit-learn pandas numpy
```

Train with default settings:
```bash
cd model
python main.py
```

Parameters are taken as arguments when calling main.py - look at `main.py` file for more details.

The best checkpoint (by val AUROC) is saved to `model/checkpoints/best_model.pt` (you can reload it later for inference or evaluation without retraining).

You can see the one I got using default settings in the ICE cluster at `/home/hice1/yli3574/MLB-project/model/checkpoints/best_model.pt`, or I can just send it to you. This runs pretty fast so you shouldn't need to use the ICE cluster tbh. Checkpoints are untracked (in .gitignore), so if anybody wants a checkpoint someone else generated you'll have to have them share it with you.

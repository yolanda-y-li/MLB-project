# MLB-Project

### Data

Below is the directory structure. You should only need to use data within the nodes and edges directories. There are different types of edges, so in case it was easier to have them separated, there is a directory that separates gene_drug_edges.tsv into three separate tsvs. 

If you want to run get_data.sh yourself, you need get_data.sh, join_gene_family.py, join_hgnc.py, and hetionet-v1.0-edges.sif.gz in the same directory and have a conda environment named ete3_env with requests and pandas installed. hetionet-v1.0-edges.sif.gz could not be curled and has to be downloaded directly from https://github.com/hetio/hetionet/blob/main/hetnet/tsv/hetionet-v1.0-edges.sif.gz. 

warning -  gene_family_edges.tsv may include gene duplicates bc some genes are in multiple families 

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



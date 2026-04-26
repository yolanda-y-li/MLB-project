import pandas as pd

# gene_families has a header but gene_nodes does not
gene_families = pd.read_csv("gene_families.tsv", sep="\t")
gene_nodes    = pd.read_csv("gene_nodes.tsv",    sep="\t", header=None)

# Left join to keep all rows of gene_nodes
# Join key on gene_families col 2 ("symbol") and gene_nodes col 2 (index 1)
merged = gene_nodes.merge(
    gene_families,
    left_on=1,
    right_on="symbol",
    how="left"
)

# make nullable integer (no decimals), missing as "NA"
merged["family_id"]    = merged["family_id"].astype("Int64")
merged["parent_fam_id"] = merged["parent_fam_id"].astype("Int64")

print(f"Output shape: {merged.shape}")
print(merged.head(10).to_string(index=False))

out_path = "gene_family_joined.tsv"
merged.to_csv(out_path, sep="\t", index=False, na_rep="NA")
print(f"\nSaved to {out_path}")

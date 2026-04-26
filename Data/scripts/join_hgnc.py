import pandas as pd

# Load all three tables
hierarchy   = pd.read_csv("hierarchy.tsv",   sep="\t")
hgnc_genes  = pd.read_csv("hgnc_genes.tsv",  sep="\t")
gene_family = pd.read_csv("gene_has_family.tsv", sep="\t")

# hgnc_genes uses "HGNC:5" format; gene_has_family uses the bare integer.
# Extract the numeric part so the keys match.
hgnc_genes["hgnc_id_int"] = hgnc_genes["hgnc_id"].str.replace("HGNC:", "", regex=False).astype(int)

# Join 1: gene_has_family and hgnc_genes
merged = gene_family.merge(
    hgnc_genes[["hgnc_id_int", "hgnc_id", "symbol", "name"]],
    left_on="hgnc_id",
    right_on="hgnc_id_int",
    how="left"
).drop(columns=["hgnc_id_int", "hgnc_id_x"])   # keep the "HGNC:5" form as hgnc_id
merged.rename(columns={"hgnc_id_y": "hgnc_id"}, inplace=True)

# Join 2: result and hierarchy for family parent info
merged = merged.merge(
    hierarchy,
    left_on="family_id",
    right_on="child_fam_id",
    how="left"
)

# Tidy column order
merged = merged[["hgnc_id", "symbol", "name", "family_id", "parent_fam_id"]]

# Convert parent_fam_id to nullable integer
merged["parent_fam_id"] = merged["parent_fam_id"].astype("Int64")

print(f"Output shape: {merged.shape}")
print(merged.head(10).to_string(index=False))

# Save
out_path = "gene_families.tsv"
merged.to_csv(out_path, sep="\t", index=False, na_rep="NA")
print(f"\nSaved to {out_path}")

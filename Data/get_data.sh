#!/bin/bash

#### set up env #####
source ~/miniforge3/etc/profile.d/conda.sh
#env with pandas, requests
conda activate ete3_env

##### download data #####

#Download data from hetnet
#curl -O https://raw.githubusercontent.com/hetio/hetionet/raw/refs/heads/main/hetnet/tsv/hetionet-v1.0-edges.sif.gz
#must download gz through ui
curl -O https://raw.githubusercontent.com/hetio/hetionet/refs/heads/main/hetnet/tsv/hetionet-v1.0-nodes.tsv

#download data from hgnc
curl -O https://storage.googleapis.com/public-download-files/hgnc/csv/csv/genefamily_db_tables/hierarchy.csv
curl -O https://storage.googleapis.com/public-download-files/hgnc/csv/csv/genefamily_db_tables/gene_has_family.csv
curl -O https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt 
curl -O https://storage.googleapis.com/public-download-files/hgnc/csv/csv/genefamily_db_tables/family.csv

#change file types
mv hgnc_complete_set.txt hgnc_complete_set.tsv
sed 's/,/\t/g' hierarchy.csv > hierarchy.tsv
sed 's/,/\t/g' gene_has_family.csv > gene_has_family.tsv
#sed 's/,/\t/g' family.csv > family.tsv
python3 -c "
import csv, sys
r = csv.reader(sys.stdin, delimiter='\t', quotechar='\"')
w = csv.writer(sys.stdout, delimiter='\t', quotechar='\"')
[w.writerow(row[:3]) for row in r]
" < family.csv > family.tsv

#### process ######

###hetnet###

#filter hetnet data to only include gene, compound, and Pharmacologic Class nodes
awk -F'\t' '$3 ~ /Gene/' hetionet-v1.0-nodes.tsv > gene_nodes.tsv
awk -F'\t' '$3 ~ /Compound/' hetionet-v1.0-nodes.tsv > drug_nodes.tsv
awk -F'\t' '$3 ~ /Pharmacologic Class/' hetionet-v1.0-nodes.tsv > drug_class_nodes.tsv


#filter hetnet data to only include edges between gene and drug nodes
gzcat hetionet-v1.0-edges.sif.gz | awk -F'\t' '$2 ~ /CbG/' > gene_drug_edges.tsv
gzcat hetionet-v1.0-edges.sif.gz | awk -F'\t' '$2 ~ /CdG/' >> gene_drug_edges.tsv
gzcat hetionet-v1.0-edges.sif.gz | awk -F'\t' '$2 ~ /CuG/' >> gene_drug_edges.tsv

#also created separate tsvs for each edge type
gzcat hetionet-v1.0-edges.sif.gz | awk -F'\t' '$2 ~ /CbG/' > gene_drug_binds_edges.tsv
gzcat hetionet-v1.0-edges.sif.gz | awk -F'\t' '$2 ~ /CdG/' > gene_drug_downreg_edges.tsv
gzcat hetionet-v1.0-edges.sif.gz | awk -F'\t' '$2 ~ /CuG/' > gene_drug_upreg_edges.tsv

#filter hetnet data to only include edges between drug and Pharmacologic class nodes
gzcat hetionet-v1.0-edges.sif.gz | awk -F'\t' '$2 ~ /PCiC/' > class_drug_edges.tsv


###hgnc###

#filter hgnc data to only include data we need
cut -f 1,2,3 hgnc_complete_set.tsv > hgnc_genes.tsv

#cut -f 1,2,3 family.tsv > gene_family_nodes.tsv
python3 -c "
import csv, sys
r = csv.reader(sys.stdin, delimiter='\t', quotechar='\"')
w = csv.writer(sys.stdout, delimiter='\t', quotechar='\"')
[w.writerow(row[:3]) for row in r]
" < family.tsv > gene_family_nodes.tsv

#makes gene_families.tsv from 
    #hierarchy.tsv
    #hgnc_genes.tsv
    #gene_has_family.tsv
python join_hgnc.py


###combine###


#makes gene_family_joined from 
    #gene_families.tsv
    #gene_nodes.tsv
python join_gene_family.py

#simplify 
cut -f 1,7,8 gene_family_joined.tsv > gene_family_edges.tsv


#organize directory
mkdir raw_data
mv family.csv ./raw_data
mv gene_has_family.csv ./raw_data
mv hetionet-v1.0-nodes.tsv ./raw_data
mv hgnc_complete_set.tsv ./raw_data
mv hetionet-v1.0-edges.sif.gz ./raw_data	
mv hierarchy.csv ./raw_data


mkdir process_data
mv family.tsv ./process_data
mv gene_families.tsv ./process_data
mv gene_has_family.tsv ./process_data
mv hgnc_genes.tsv ./process_data
mv hierarchy.tsv ./process_data


mkdir edges
mv gene_drug_edges.tsv ./edges
mv class_drug_edges.tsv ./edges
mv gene_family_edges.tsv ./edges


mkdir ./edges/gene_drug_by_edge_type
mv gene_drug_binds_edges.tsv ./edges/gene_drug_by_edge_type
mv gene_drug_downreg_edges.tsv ./edges/gene_drug_by_edge_type
mv gene_drug_upreg_edges.tsv ./edges/gene_drug_by_edge_type

mkdir nodes
mv drug_class_nodes.tsv ./nodes
mv drug_nodes.tsv ./nodes
mv gene_family_nodes.tsv ./nodes
mv gene_nodes.tsv ./nodes

mkdir scripts
mv join_gene_family.py ./scripts
mv join_gene_nodes.py ./scripts
mv join_hgnc.py ./scripts
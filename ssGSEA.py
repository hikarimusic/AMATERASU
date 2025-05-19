# -------------------------

geneset_prefix_remove = 'HALLMARK_'
geneset_combine_mode = 'add'
ssGSEA_weight = 0.75

# -------------------------

import sys
import os
import pandas as pd
import numpy as np

def calculate_ssGSEA_score(expression_values, gene_list, gene_set2, weight=ssGSEA_weight):
    gene_names = expression_values.index
    n_rows = len(gene_names)
    
    # Z values normalization
    if weight == 0:
        ranked_expression = np.ones(n_rows)
    else:
        x = expression_values[gene_list]
        ranked_expression = (x - np.mean(x)) / np.std(x, ddof=1)
    
    # Calculate tag indicator
    tag_indicator = np.sign(np.isin(gene_list, gene_set2).astype(int))
    N = len(gene_list)
    Nh = len(gene_set2)
    Nm = N - Nh
    ind = np.where(tag_indicator == 1)[0]
    ranked_expression = np.abs(ranked_expression[ind]) ** weight 

    # Calculate up and down
    sum_ranked_expression = np.sum(ranked_expression)
    up = ranked_expression / sum_ranked_expression
    gaps = np.append(ind-1, N-1) - np.append(-1, ind)
    down = gaps / Nm
    
    # Calculate running sum
    RES = np.cumsum(np.append(up, up[Nh-1]) - down)
    valleys = RES[:Nh] - up
    
    # Calculate final ES
    gaps = gaps + 1
    RES = np.append(valleys, 0) * gaps + 0.5 * (np.append(0, RES[:Nh]) - np.append(valleys, 0)) * gaps
    ES = np.sum(RES)
    
    return ES

def ssGSEA(summarize_file, gmt_file):
    # Read data
    print("[Read Data] ...", end='\r')
    df = pd.read_csv(summarize_file, low_memory=False)
    start_gene_index = df.columns.get_loc('START_GENE')
    metadata = df.iloc[:, :start_gene_index + 1]
    gene_data = df.iloc[:, start_gene_index + 1:]
    print("[Read Data] Complete                 ")
    
    # Read gene sets
    print("[Load GeneSets] ...", end='\r')
    gene_sets = {}
    with open(gmt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            gene_set_name = parts[0]
            if gene_set_name.startswith(geneset_prefix_remove):
                gene_set_name = gene_set_name[len(geneset_prefix_remove):]
            genes = list(parts[2:])
            gene_sets[gene_set_name] = genes
    print(f"[Load GeneSets] {len(gene_sets)}                 ")

    # Pre-compute gene set indices for each gene set
    gene_names = gene_data.columns
    gene_set_indices = {}
    for gene_set_name, gene_set in gene_sets.items():
        gene_set_indices[gene_set_name] = np.where(np.isin(gene_names, gene_set))[0]

    # Calculate enrichment scores
    print("[Perform ssGSEA] ...", end='\r')
    enrichment_scores = pd.DataFrame(index=df.index, columns=gene_sets.keys())
    
    total_samples = len(df)
    for idx, sample in enumerate(df.index, 1):
        sample_expression = gene_data.loc[sample]

        # Pre-compute sorted gene list for this sample
        pre_sorted_indices = np.argsort(-sample_expression.values, kind='stable')
        
        for gene_set_name, gene_set in gene_sets.items():
            score = calculate_ssGSEA_score(sample_expression, pre_sorted_indices, gene_set_indices[gene_set_name])
            enrichment_scores.loc[sample, gene_set_name] = score
            
        print(f"[Perform ssGSEA] {idx}/{total_samples}", end='\r')
    
    print("[Perform ssGSEA] Complete                 ")

    # Combine UP/DN
    if geneset_combine_mode in ["add", "replace"]:
        print("[Combine UP/DN] ...", end='\r')
        dn_columns = [col for col in enrichment_scores.columns if col.endswith('_DN')]
        for dn_col in dn_columns:
            up_col = dn_col.replace('_DN', '_UP')
            if up_col in enrichment_scores.columns:
                base_name = dn_col.replace('_DN', '')
                combined_score = enrichment_scores[up_col] - enrichment_scores[dn_col]
                dn_idx = enrichment_scores.columns.get_loc(dn_col)
                enrichment_scores.insert(dn_idx + 1, base_name, combined_score)
                if geneset_combine_mode == "replace":
                    enrichment_scores = enrichment_scores.drop(columns=[up_col, dn_col])
                print(f"[Combine UP/DN] {base_name}                 ")

    # Save results
    print("[Save Results] ...", end='\r')
    result = pd.concat([metadata, enrichment_scores], axis=1)
    output_dir = os.path.dirname(summarize_file)
    output_file = os.path.join(output_dir, 'summary_ssGSEA.csv')
    result.to_csv(output_file, index=False)
    print("[Save Results] Complete                 ")

if __name__ == "__main__":
    '''Command: python3 ssGSEA.py <cohort/summary.csv> <geneset.gmt>'''
    summarize_file = sys.argv[1]
    gmt_file = sys.argv[2]
    
    ssGSEA(summarize_file, gmt_file)
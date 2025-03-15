# -------------------------

pca_plot_format = 'png'
pca_plot_size = (3.5, 3.5)
pca_plot_dpi = 600
pca_plot_fontsize = 6
pca_plot_dotsize = 10
pca_plot_color = 'deep'
pca_plot_labels = False
pca_plot_group_order = {} # Example: {"Gender": ['female', 'male']}

gene_threshold = 1
gene_normalize = 'median' # 'median', 'mean', none

cluster_gene_metric = 'seuclidean'
cluster_case_metric = 'seuclidean'
cluster_gene_method = 'ward'
cluster_case_method = 'ward'
cluster_column_name = 'nClusters'

heatmap_format = 'png'
heatmap_size = (7.0, 7.0)
heatmap_dpi = 600
heatmap_font_size = 6
heatmap_color = 'seismic'
heatmap_group_preset = ["Set1", "tab10", "Dark2"]
heatmap_group_order = {} # Example: {"Gender": ['female', 'male']}
heatmap_gene_name = False

# -------------------------

import sys
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

def generate_pca_plot(metadata, gene_data, group_columns, output_dir):
    print("[PCA] ...    ", end='\r')
    
    # Normalize
    valid_samples = ~metadata[group_columns].isna().any(axis=1)
    gene_data = gene_data[valid_samples].reset_index(drop=True)
    metadata = metadata[valid_samples].reset_index(drop=True)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(gene_data)

    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_features)
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
    labels = metadata[group_columns]
    sample_ids = metadata.iloc[:, 0]
    pca_df = pd.concat([pca_df, labels, sample_ids], axis=1)

    # Create plots
    plt.style.use('ggplot')
    for group_column in group_columns:
        plt.figure(figsize=pca_plot_size)
        if group_column in pca_plot_group_order:
            hue_order = pca_plot_group_order[group_column]
        else:
            hue_order = sorted(metadata[group_column].unique(), key=str)
        sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue=group_column, palette=pca_plot_color, s=pca_plot_dotsize, hue_order=hue_order)
       
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=pca_plot_fontsize)
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=pca_plot_fontsize)
        plt.xticks(fontsize=pca_plot_fontsize)
        plt.yticks(fontsize=pca_plot_fontsize)
        plt.legend(title=group_column, title_fontsize=pca_plot_fontsize, fontsize=pca_plot_fontsize)

        # Add sample ID labels
        if pca_plot_labels == True:
            for idx, row in pca_df.iterrows():
                plt.annotate(row[sample_ids.name], (row['PC1'], row['PC2']), xytext=(3, 3), textcoords='offset points', fontsize=5, alpha=0.7)

        output_file = os.path.join(output_dir, f'cluster_PCA_{group_column.replace(" ", "_")}.' + pca_plot_format)
        plt.savefig(output_file, format=pca_plot_format, dpi=pca_plot_dpi, bbox_inches='tight')
        plt.close()
   
    print(f"[PCA] Complete                 ")

def cluster(summarize_file, group_columns):
    # Read data
    print(f"[Read Data] ...", end='\r')  
    df = pd.read_csv(summarize_file, low_memory=False)
    start_gene_index = df.columns.get_loc('START_GENE')
    metadata = df.iloc[:, :start_gene_index + 1]
    gene_data = df.iloc[:, start_gene_index + 1:]
    print("[Read Data] Complete                 ")

    # Remove cluster number
    n_clusters = None
    for column in group_columns:
        if column.startswith('-'):
            n_clusters = int(column[1:])
            group_columns.remove(column)

    # PCA
    generate_pca_plot(metadata, gene_data, group_columns, os.path.dirname(summarize_file))

    # Filter and normalize genes
    print(f"[Filter Genes] ...", end='\r')
    expressed_genes = gene_data.columns[gene_data.mean() > gene_threshold]
    gene_data = gene_data[expressed_genes]
    high_var_genes = gene_data.columns[gene_data.var() > 0]
    selected_gene_data = gene_data[high_var_genes]
    if gene_normalize == 'median':
        target_median = selected_gene_data.median(axis=1).median(axis=0)
        scale_factors = target_median / selected_gene_data.median(axis=1)
    elif gene_normalize == 'mean':
        target_median = selected_gene_data.mean(axis=1).mean(axis=0)
        scale_factors = target_median / selected_gene_data.mean(axis=1)
    else:
        scale_factors = 1
    expression_data = selected_gene_data.multiply(scale_factors, axis=0)
    print(f"[Filter Genes] {expression_data.shape[1]}                 ")

    # Hierarchy clustering (using z-scores for gene clustering)
    print(f"[Hierarchy Cluster] ...", end='\r')
    # Z-score normalization before clustering
    scaler = StandardScaler()
    z_scores_for_clustering = pd.DataFrame(
        scaler.fit_transform(expression_data),
        columns=expression_data.columns,
        index=expression_data.index
    )

    gene_dist = pdist(z_scores_for_clustering.T, metric=cluster_gene_metric)
    case_dist = pdist(expression_data, metric=cluster_case_metric)
    
    gene_linkage = hierarchy.linkage(gene_dist, method=cluster_gene_method)
    case_linkage = hierarchy.linkage(case_dist, method=cluster_case_method)
    case_order = hierarchy.leaves_list(case_linkage)
    gene_order = hierarchy.leaves_list(gene_linkage)
    data_ordered = expression_data.iloc[case_order, gene_order]

    # Output clustering results
    if n_clusters is not None:
        cluster_labels = hierarchy.fcluster(case_linkage, t=n_clusters, criterion='maxclust')
        output_dir = os.path.dirname(summarize_file)
        
        summary_with_clusters = df.copy()
        start_gene_idx = summary_with_clusters.columns.get_loc('START_GENE')
        summary_with_clusters.insert(start_gene_idx, cluster_column_name, [f"Cluster{label}" for label in cluster_labels])
        clustered_summary_output = os.path.join(output_dir, 'summary_cluster.csv')
        summary_with_clusters.to_csv(clustered_summary_output, index=False)

        start_gene_idx = summary_with_clusters.columns.get_loc('START_GENE')
        cohort_with_clusters = summary_with_clusters.iloc[:, :start_gene_idx]
        clustered_cohort_output = os.path.join(output_dir, 'cohort_cluster.csv')
        cohort_with_clusters.to_csv(clustered_cohort_output, index=False)

    print("[Hierarchy Cluster] Complete                 ")
    
    # Create heatmap
    print("[Create Heatmap] ...", end='\r')
    # Calculate z-scores for the ordered data
    z_scores = pd.DataFrame(
        scaler.fit_transform(data_ordered), 
        columns=data_ordered.columns, 
        index=data_ordered.index
    )
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=heatmap_size)
    gs = fig.add_gridspec(2 + len(group_columns), 3, 
                        width_ratios=[1, 10, 1.5], 
                        height_ratios=[1] + [0.1] * len(group_columns) + [10],
                        left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.02, hspace=0.02)
    
    # Heatmap - FIX: flip the gene order to match dendrogram orientation
    ax_heatmap = fig.add_subplot(gs[1+len(group_columns), 1])
    vmin = np.percentile(z_scores.values, 1)
    vmax = np.percentile(z_scores.values, 99)
    sns.heatmap(z_scores.T.iloc[::-1], cmap=heatmap_color, center=0, vmin=vmin, vmax=vmax,
                xticklabels=False, yticklabels=heatmap_gene_name, cbar=False, ax=ax_heatmap)

    # Gene dendrogram
    if heatmap_gene_name == True:
        # If showing gene names, need to reverse order to match the flipped heatmap
        ax_heatmap.set_yticklabels(data_ordered.columns[::-1], fontsize=heatmap_font_size, rotation=0)
    else:
        ax_gene_dendrogram = fig.add_subplot(gs[1+len(group_columns), 0])
        hierarchy.dendrogram(gene_linkage, orientation='left', ax=ax_gene_dendrogram, link_color_func=lambda k: 'black')
        ax_gene_dendrogram.axis('off')

    # Case dendrogram
    ax_case_dendrogram = fig.add_subplot(gs[0, 1])
    hierarchy.dendrogram(case_linkage, ax=ax_case_dendrogram, link_color_func=lambda k: 'black')
    ax_case_dendrogram.axis('off')

    # Group indicators
    color_preset = heatmap_group_preset
    n_preset = len(color_preset)
    for i, group_column in enumerate(group_columns):
        ax_groups = fig.add_subplot(gs[1+i, 1], sharex=ax_heatmap)
        ax_groups.set_ylim(0, 1)
        
        if group_column in heatmap_group_order:
            unique_groups = heatmap_group_order[group_column]
        else:
            unique_groups = sorted(metadata[group_column].dropna().unique(), key=str)
        color_palette = sns.color_palette(color_preset[i%n_preset], n_colors=len(unique_groups))
        color_map = dict(zip(unique_groups, color_palette))
        
        for j, sample in enumerate(data_ordered.index):
            group = metadata.loc[sample, group_column]
            if pd.isna(group):
                ax_groups.axvspan(j, j+1, facecolor='white', alpha=1)
            else:
                ax_groups.axvspan(j, j+1, facecolor=color_map[group], alpha=1)
        
        ax_groups.set_xlim(0, len(data_ordered))
        ax_groups.axis('off')
    
    # Legend and colorbar
    ax_legend = fig.add_subplot(gs[1+len(group_columns), 2])
   
    legend_elements = []
    for i, group_column in enumerate(group_columns):
        if group_column in heatmap_group_order:
            unique_groups = heatmap_group_order[group_column]
        else:
            unique_groups = sorted(metadata[group_column].dropna().unique(), key=str)
        color_palette = sns.color_palette(color_preset[i%n_preset], n_colors=len(unique_groups))
        color_map = dict(zip(unique_groups, color_palette))
        legend_elements.extend([Rectangle((0, 0), 0.5, 0.5, facecolor="white", label=group_column)])
        legend_elements.extend([Rectangle((0, 0), 0.5, 0.5, facecolor=color_map[group], label=group) for group in unique_groups])
        legend_elements.extend([Rectangle((0, 0), 0.5, 0.5, facecolor="white", label="") for _ in range(2)])

    cmap = plt.get_cmap(heatmap_color)
    z_min = round(np.floor(vmin / 0.2))
    z_max = round(np.ceil(vmax / 0.2))
    z_values = list(range(z_max, z_min, -1))
    cbar_elements = []
    for z in z_values:
        if z % 5 == 0:
            label = f'{round(z*0.2)}'
        else:
            label = ''
        cbar_elements.append(Rectangle((0, 0), 0.5, 0.5, facecolor=cmap(z*0.1/max(abs(vmax), abs(vmin))+0.5), label=label))
    
    all_elements = legend_elements + cbar_elements
    legend = ax_legend.legend(handles=all_elements, loc='center', 
                              ncol=1, handlelength=1, handleheight=1, 
                              handletextpad=0.5, columnspacing=0.5, labelspacing=0.0,
                              prop={'size': heatmap_font_size})

    for text in legend.get_texts():
        if text.get_text() in group_columns:
            text.set_weight('bold')
    legend.get_frame().set_linewidth(0.0)
    ax_legend.axis('off')
    print("[Create Heatmap] Complete                 ")
    
    # Save the plot
    print("[Save Heatmap] ...", end='\r')
    output_dir = os.path.dirname(summarize_file)
    output_file = os.path.join(output_dir, f'cluster_heatmap_{"_".join(group_columns).replace(" ", "_")}.' + heatmap_format)
    plt.savefig(output_file, format=heatmap_format, dpi=heatmap_dpi, bbox_inches='tight')
    plt.close()
    
    print("[Save Heatmap] Complete                 ")

if __name__ == "__main__":
    '''Command: python3 cluster.py <cohort/summary.csv> <group_column1> <group_column2> ... '''
    summarize_file = sys.argv[1]
    group_columns = sys.argv[2:]

    cluster(summarize_file, group_columns)
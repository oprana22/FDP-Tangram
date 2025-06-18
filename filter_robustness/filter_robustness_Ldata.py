import scanpy as sc
import squidpy as sq
import numpy as np
import pandas as pd
from anndata import AnnData
import pathlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import skimage
import seaborn as sns
import tangram as tg
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import mean_squared_error
import os

def test_f_robustness(adata_sc, adata_st, target_count, n_iter=10, n_epochs=500, device="cpu"):
    """
    Function runs tg n_iter times on the same data and returns np array with all the filter values.
    We want to see if the same cells are chosen by the filter.
    pp_adatas needs to be run before calling this function.
    """

    f_ls = np.zeros((n_iter, adata_sc.shape[0]))  # np array where all computed filters will be stored.
    
    for i in range(n_iter):
        print(f"Executing Tangram iter {i}")
        ad_map = tg.map_cells_to_space(adata_sc, adata_st,
            mode="constrained",  
            density_prior='rna_count_based',
            target_count=target_count,
            lambda_d = 0.89,
            lambda_g2 = 0.99,
            num_epochs=n_epochs,
            device=device,
            )
        
        f_ls[i,:]=ad_map.obs["F_out"].values
        
    return np.vstack(f_ls)

os.makedirs('results', exist_ok=True)

adata_st = sc.read('../lucas_data/Visium_Mouse_Brain_SPAPROS_filtered_celltypes_annotated.h5ad')

adata_sc = sc.read('../lucas_data/SC_REF_for_VISIUM_preprocessed.h5ad')

tg.pp_adatas(adata_sc, adata_st, genes=None) #prepare for mapping.

filter_results = test_f_robustness(adata_sc, adata_st, target_count=5000)

# Compute metrics
cos_sim = cosine_similarity(filter_results)
eucl_dist = euclidean_distances(filter_results)
mean_row = filter_results.mean(axis=0)
mse = np.mean([mean_squared_error(row, mean_row) for row in filter_results])

# Get summary statistics
cos_sim_mean = np.mean(cos_sim[np.triu_indices_from(cos_sim, k=1)])  # exclude diagonal
eucl_dist_mean = np.mean(eucl_dist[np.triu_indices_from(eucl_dist, k=1)])

# === Create and write metrics to text file
metrics_filename = 'results/similarity_metrics.txt'
with open(metrics_filename, 'w') as f:
    f.write("=== Similarity Metrics Report ===\n")
    f.write(f"Mean Cosine Similarity (off-diagonal): {cos_sim_mean:.4f}\n")
    f.write(f"Mean Euclidean Distance (off-diagonal): {eucl_dist_mean:.4f}\n")
    f.write(f"Mean Squared Error (MSE to mean row): {mse:.4f}\n")

print(f"Similarity metrics written to: {metrics_filename}")

# === Save filter_results to CSV
filter_results_df = pd.DataFrame(filter_results)

filter_results_filename = 'results/filter_results.csv'
filter_results_df.to_csv(filter_results_filename, index=False)

print(f"Filter results written to: {filter_results_filename}")


# ----- 1. Cosine Similarity -----
plt.figure(figsize=(10, 8))
sns.heatmap(cos_sim, cmap='viridis', xticklabels=False, yticklabels=False)
plt.title('Cosine Similarity Between Rows')
plt.xlabel('Row Index')
plt.ylabel('Row Index')
plt.tight_layout()
plt.savefig('results/similarity_f_consistency.png')
plt.show()

# ----- 2. Euclidean Distance -----
plt.figure(figsize=(10, 8))
sns.heatmap(eucl_dist, cmap='magma_r', xticklabels=False, yticklabels=False)
plt.title('Euclidean Distance Between Rows')
plt.xlabel('Row Index')
plt.ylabel('Row Index')
plt.tight_layout()
plt.savefig('results/eucliandist_f_consistency.png')
plt.show()

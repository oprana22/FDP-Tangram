import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import squidpy as sq

import cell2location
from cell2location.utils.filtering import filter_genes
from cell2location.models import RegressionModel, Cell2location
from matplotlib import rcParams

# Load spatial data

adata_st = sq.datasets.visium_fluo_adata_crop()
adata_st = adata_st[
    adata_st.obs.cluster.isin([f"Cortex_{i}" for i in np.arange(1, 5)])
].copy()

adata_st.X = adata_st.raw.X.copy()

# Load single-cell reference data
adata_sc = sq.datasets.sc_mouse_cortex()
adata_sc.X = adata_sc.raw.X.copy()


# Paths for saving results
results_folder = './results'
ref_run_name = f'{results_folder}/reference_signatures'
run_name = f'{results_folder}/cell2location_map'

# Clean up gene names in spatial data
adata_st.var['SYMBOL'] = adata_st.var_names
adata_st.var['MT_gene'] = [gene.startswith('MT-') for gene in adata_st.var['SYMBOL']]
adata_st.obsm['MT'] = adata_st[:, adata_st.var['MT_gene'].values].X.toarray()
adata_st = adata_st[:, ~adata_st.var['MT_gene'].values]

# Clean up gene names in sc data
adata_sc.var['SYMBOL'] = adata_sc.var.index

# Filter lowly expressed genes in single-cell data
selected = filter_genes(adata_sc, cell_count_cutoff=5, cell_percentage_cutoff2=0.03, nonz_mean_cutoff=1.12)
adata_sc = adata_sc[:, selected].copy()

# Set up for regression model
RegressionModel.setup_anndata(adata=adata_sc, labels_key='cell_subclass')

# Create and train regression model
mod = RegressionModel(adata_sc)
mod.view_anndata_setup()
mod.train(max_epochs=500)

# Plot training history
mod.plot_history(0)
plt.legend(labels=['full data training'])

# Export reference signatures
adata_sc = mod.export_posterior(
    adata_sc, sample_kwargs={'num_samples': 1000, 'batch_size': mod.adata.n_obs}
)
adata_sc.write(f'{ref_run_name}/sc.h5ad')
mod.save(f'{ref_run_name}', overwrite=True)

shared_genes = adata_st.var_names.intersection(adata_sc.varm['means_per_cluster_mu_fg'].index)

# Subset and align both AnnData objects to shared genes
adata_st = adata_st[:, shared_genes].copy()
cell_state_df = adata_sc.varm['means_per_cluster_mu_fg'].loc[shared_genes]

# Prepare spatial data for cell2location model
Cell2location.setup_anndata(adata_st)

# Run Cell2location model using reference signatures
c2l_model = Cell2location(
    adata_st,
    cell_state_df=cell_state_df,
    N_cells_per_location=30,
    detection_alpha=1,
    detection_mean=20
)
c2l_model.train(max_epochs=30000)


# Export cell abundance estimates (posterior)
adata_st = c2l_model.export_posterior(
    adata_st, sample_kwargs={'num_samples': 1000, 'batch_size': adata_st.n_obs}
)

# Save model and results
c2l_model.save(f"{run_name}", overwrite=True)
adata_st.write(f"{run_name}/sp.h5ad")

print("Pipeline complete. Results saved to:", run_name)


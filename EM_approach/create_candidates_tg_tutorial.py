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

def create_candidates(adata_sc, adata_st, n_iter, p_thresh):
    
    candidates = np.zeros(shape = (adata_sc.shape[0], adata_st.shape[0]), dtype="int") #matrix to store candidates
    
    for _ in range(n_iter):
        ad_map = tg.map_cells_to_space(adata_sc,
                        adata_st,
                        mode="cells",  
                        density_prior='rna_count_based',
                        lambda_d = 0.89,
                        lambda_g2 = 0.99,
                        num_epochs=350,
                        device="cpu",
                        )
        
        candidates = np.add(candidates, (ad_map.X > p_thresh).astype(int))
    
    return (candidates > 0).astype(int)

#-------------- load data -------------
adata_st = sq.datasets.visium_fluo_adata_crop()
adata_st = adata_st[
    adata_st.obs.cluster.isin([f"Cortex_{i}" for i in np.arange(1, 5)])
].copy()

adata_sc = sq.datasets.sc_mouse_cortex()
#-------------- prepare for mapping --------------
sc.tl.rank_genes_groups(adata_sc, groupby="cell_subclass", use_raw=False)
markers_df = pd.DataFrame(adata_sc.uns["rank_genes_groups"]["names"]).iloc[0:100, :]
markers = list(np.unique(markers_df.melt().value.values))
tg.pp_adatas(adata_sc, adata_st, genes=markers)

#-------------- get candidates for EM ------------
candidates = create_candidates(adata_sc, adata_st, n_iter = 20, p_thresh = 0.9)
np.save("candidates/candidates_90_tg.npy", candidates)

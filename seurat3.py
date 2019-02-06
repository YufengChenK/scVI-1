#!/usr/bin/env python
# coding: utf-8

# *First compiled on May 5, 2017.*

# # Clustering 3k PBMCs following a Seurat Tutorial

# Scanpy allows to reproduce most of Seurat's ([Satija *et al.*, 2015](https://doi.org/10.1038/nbt.3192)) standard clustering tutorial as described on http://satijalab.org/seurat/pbmc3k_tutorial.html (July 26, 2017). We gratefully acknowledge  the Seurat authors for publishing the tutorial!
# 
# The data consists in *3k PBMCs from a Healthy Donor* and is freely available from 10x Genomics ([here](http://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz) from this [webpage](https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/pbmc3k)).

# In[4]:
from IPython import get_ipython

get_ipython().run_line_magic('matplotlib', 'qt')
import numpy as np
import scanpy.api as sc

sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.set_figure_params(dpi=80)  # low dpi (dots per inch) yields small inline figures
sc.logging.print_version_and_date()
sc.logging.print_versions_dependencies_numerics()
results_file = './write/pmbc3k.h5ad'


# In[1]:


path = './data/filtered_gene_bc_matrices/hg19/'
adata = sc.read(path + 'matrix.mtx', cache=True).transpose()
adata.var_names = np.genfromtxt(path + 'genes.tsv', dtype=str)[:, 1]
adata.obs_names = np.genfromtxt(path + 'barcodes.tsv', dtype=str)


# In[2]:


adata.var_names_make_unique()


# ## Preprocessing

# Basic filtering.

# In[4]:


sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)


# Plot some information about mitochondrial genes, important for quality control

# In[5]:


mito_genes = [name for name in adata.var_names if name.startswith('MT-')]
# for each cell compute fraction of counts in mito genes vs. all genes
# the ".A1" is only necessary, as X is sparse - it transform to a dense array after summing
adata.obs['percent_mito'] = np.sum(
    adata[:, mito_genes].X, axis=1).A1 / np.sum(adata.X, axis=1).A1
# add the total counts per cell as observations-annotation to adata
adata.obs['n_counts'] = np.sum(adata.X, axis=1).A1


# A violin plot of the computed quality measures.

# In[7]:


sc.pl.violin(adata, ['n_genes', 'n_counts', 'percent_mito'],
             jitter=0.4, multi_panel=True)


# Remove cells that have too many mitochondrial genes expressed or too many total counts.

# In[8]:


sc.pl.scatter(adata, x='n_counts', y='percent_mito')
sc.pl.scatter(adata, x='n_counts', y='n_genes')


# Actually do the filtering.

# In[9]:


adata = adata[adata.obs['n_genes'] < 2500, :]
adata = adata[adata.obs['percent_mito'] < 0.05, :]


# Set the `.raw` attribute of AnnData object to the logarithmized raw gene expression for later use in differential testing and visualizations of gene expression. This simply freezes the state of the data stored in `adata_raw`.

# In[10]:


adata_raw = sc.pp.log1p(adata, copy=True)
adata.raw = adata_raw


# Per-cell normalize the data matrix $\mathbf{X}$, identify highly-variable genes and compute logarithm.

# In[11]:


sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
filter_result = sc.pp.filter_genes_dispersion(
    adata.X, min_mean=0.0125, max_mean=3, min_disp=0.5)
sc.pl.filter_genes_dispersion(filter_result)


# Actually do the filtering.

# In[12]:


adata = adata[:, filter_result.gene_subset]


# Logarithmize the data.

# In[13]:


sc.pp.log1p(adata)


# Regress out effects of total counts per cell and the percentage of mitochondrial genes expressed. Scale the data to unit variance.

# In[17]:


sc.pp.regress_out(adata, ['n_counts', 'percent_mito'])


# In[14]:


sc.pp.scale(adata, max_value=10)


# Save the result.

# In[15]:


adata.write(results_file)


# In[16]:


print(adata)


# ## PCA

# Compute PCA and make a scatter plot.

# In[17]:


sc.tl.pca(adata)


# In[18]:


adata.obsm['X_pca'] *= -1  # multiply by -1 to match Seurat R
sc.pl.pca_scatter(adata, color='CST3')


# Let us inspect the contribution of single PCs to the total variance in the data. This gives us information about how many PCs we should consider in order to compute the neighborhood relations of cells, e.g. used in the clustering function  `sc.tl.louvain()` or tSNE `sc.tl.tsne()`. In our experience, often, a rough estimate of the number of PCs does fine. Seurat provides many more functions, here.

# In[19]:


sc.pl.pca_variance_ratio(adata, log=True)


# In[20]:


print(adata)


# ## tSNE

# In[21]:


adata = sc.read(results_file)


# In[22]:


sc.tl.tsne(adata, n_pcs=10, random_state=2)
adata.write(results_file)


# In[23]:


ax = sc.pl.tsne(adata, color=['CST3', 'NKG7'], color_map='RdBu_r')


# As we set the `.raw` attribute of AnnData (a "frozen" state of the object at a point in the pipeline where we deemed the data "raw"), the previous plots showed the raw gene expression.
# 
# By setting `use_raw` to `False`, we can also plot the scaled and corrected expression.

# In[24]:


sc.pl.tsne(adata, color=['CST3', 'NKG7'], color_map='RdBu_r', use_raw=False)


# ## Clustering

# As Seurat and many others, we use the Louvain graph-clustering method (community detection based on optimizing modularity). It has been proposed for single-cell data by [Levine *et al.* (2015)](https://doi.org/10.1016/j.cell.2015.05.04).

# In[25]:


adata = sc.read(results_file)


# In[26]:


print(adata)


# In[27]:


sc.tl.louvain(adata, n_neighbors=10, resolution=1, recompute_graph=True)


# Plot the data with tSNE. Coloring according to clustering. Clusters agree quite well with the result of Seurat.

# In[ ]:


sc.pl.tsne(adata, color='louvain_groups')


# Save this, in case we need it later.

# In[ ]:


adata.write(results_file)


# ## Finding marker genes

# Let us compute a ranking for the highly differential genes in each cluster. Here, we simply rank genes by z-score, this agrees quite well with the more advanced tests of Seurat.
# 
# For this, by default, the `.raw` attribute of AnnData is used in case it has been initialized before.

# In[ ]:


adata = sc.read(results_file)
sc.tl.rank_genes_groups(adata, 'louvain_groups')
sc.pl.rank_genes_groups(adata, n_genes=20, save='.pdf')
adata.write(results_file)


# Show the 20 top ranked genes per cluster 0, 1, ..., 7 in a dataframe.

# In[ ]:


import pandas as pd
pd.DataFrame(adata.uns['rank_genes_groups_gene_names']).loc[:20]


# Even though the ranking by z-scores is a very simple procedure, the resulting genes agree very well with the marker genes fround by Seurat. With the exception of the marker genes of group 4, all marker genes mentioned in the [Seurat Tutorial](http://satijalab.org/seurat/pbmc3k_tutorial.html) can be found the rankings, and one can hence identify the cell types.
# 
# Louvain Group | Markers | Cell Type
# ---|---|---
# 0 | IL7R | 0
# 1 | CD14, LYZ | 1
# 2 | MS4A1 |	B cells
# 3 |	CD8A |	CD8 T cells
# 4 |	FCGR3A, MS4A7 |	FCGR3A+ Monocytes
# 5 |	GNLY, NKG7 | 	NK cells
# 6 |	FCER1A, CST3 |	Dendritic Cells
# 7 |	PPBP |	Megakaryocytes

# Compare to a single cluster. 

# In[ ]:


adata = sc.read(results_file)
sc.tl.rank_genes_groups(adata, 'louvain_groups', groups=['0'], reference='1')
sc.pl.rank_genes_groups(adata, groups=['0'], n_genes=20)


# If we want a more detailed view for a certain group, use `sc.pl.rank_genes_groups_violin`.

# In[ ]:


sc.pl.rank_genes_groups_violin(adata, groups='0', n_genes=8)


# In[ ]:


adata = sc.read(results_file)
sc.pl.rank_genes_groups_violin(adata, groups='0', n_genes=8)


# If you want to compare a certain gene across groups, use the following.

# In[ ]:


sc.pl.violin(adata, 'NKG7', group_by='louvain_groups')


# Actually mark the cell types.

# In[ ]:


adata = sc.read(results_file)
adata.obs['louvain_groups'].cat.categories = [
    'CD4 T cells', 'CD14+ Monocytes',
    'B cells', 'CD8 T cells', 
    'NK cells', 'FCGR3A+ Monocytes',
    'Dendritic cells', 'Megakaryocytes']
adata.write(results_file)


# In[ ]:


adata = sc.read(results_file)
sc.pl.tsne(adata, size=10,
           legend_fontsize=12, legend_fontweight='bold',
           color='louvain_groups',
           legend_loc='on data')


# ## Saving or exporting the results

# Write the results using compression to save diskspace.

# In[ ]:


adata.write(results_file)


# If you want to export to "csv", you will usually use pandas.

# In[ ]:


# Export single fields of the annotation of observations
adata.obs[['n_counts', 'louvain_groups']].to_csv(
    './write/pbmc3k_corrected_louvain_groups.csv')


# In[ ]:


# Export single columns of the multidimensional annotation
adata.obsm.to_df()[['X_pca1', 'X_pca2']].to_csv(
    './write/pbmc3k_corrected_X_pca.csv')


# In[ ]:


# Or export everything except the data using `.write_csvs`.
adata.write_csvs(results_file[:-5])
adata


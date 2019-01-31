import scanpy.api as sc
from anndata import AnnData
from scanpy.plotting.tools.scatterplots import plot_scatter
import matplotlib.pyplot as plt
from scvi.dataset import GeneExpressionDataset
from scvi.inference import Posterior
from scvi.api.tool.dataset2scanpyObject import dataset2adata
from scvi.api.tool.array2geneDataset import XGeneDataset
from scvi.api.tool.array2adata import array2adata
import numpy as np

def posterior2adata(posterior: Posterior, _type="cell_order", **kwargs):
    latent, batch_indices, labels = posterior.get_latent()

    tsne = posterior.apply_t_sne(latent, None)
    imputation = posterior.imputation()
    table = posterior.differential_expression_table(select=None)

    gene_dataset: GeneExpressionDataset = posterior.gene_dataset
    indices = posterior.indices
    X = gene_dataset.X[indices]
    gene_names = gene_dataset.gene_names
    labels = gene_dataset.labels[indices]
    adata = array2adata(X, gene_names=gene_names, cell_types=gene_dataset.cell_types,cell_names=labels)
    adata:AnnData = adata.transpose()
    adata.obsm['X_scVI'] = latent
    adata.obsm['X_tsne'] = tsne[0]
    adata.uns["imputation"] = imputation
    dfadata = AnnData(X=table[1])
    dfadata.obs['cell_names'] = table[0]
    adata.uns["differential_expression"] = dfadata

    return adata
if __name__ == '__main__':
    pass

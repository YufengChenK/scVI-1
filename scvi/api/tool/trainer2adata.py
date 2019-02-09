import numpy as np
from anndata import AnnData

from scvi.api import array2adata
from scvi.dataset import GeneExpressionDataset


def trainer2adata(trainer, _type="cell_order"):
    latent = trainer.get_all_latent_and_imputed_values()['latent']
    gene_dataset: GeneExpressionDataset = trainer.gene_dataset
    X = gene_dataset.X
    gene_names = gene_dataset.gene_names
    labels = gene_dataset.labels
    adata = array2adata(X, gene_names=gene_names, cell_types=gene_dataset.cell_types, cell_names=labels)

    adata: AnnData = adata.transpose()
    adata.obsm['X_scVI'] = latent
    return adata

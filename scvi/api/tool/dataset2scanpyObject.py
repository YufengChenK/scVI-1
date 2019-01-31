from scvi.api.tool.get_unique_cell_names import get_unique_cell_names
from scvi.dataset import GeneExpressionDataset, CortexDataset
from anndata import AnnData
import numpy as np
import pandas as pd

def dataset2adata(gene_dataset: GeneExpressionDataset, _type="cell_order", **kwargs):
    """

    :param gene_dataset:
    :param _type: choice:["cell_order", "type_order"]
    :param kwargs:
    """
    X = gene_dataset.X
    labels = gene_dataset.labels
    cell_types = gene_dataset.cell_types
    n_labels = gene_dataset.n_labels
    gene_names = gene_dataset.gene_names
    cell_names = get_unique_cell_names(_type, cell_types, labels)
    n_genes = pd.Series(np.sum(gene_dataset.X, axis=0), index=gene_names)
    n_cells = pd.Series(np.sum(gene_dataset.X, axis=1), index=cell_names)
    X = np.transpose(X)

    adata = AnnData(X)
    adata.obs_names = gene_names
    adata.var_names = cell_names
    adata.obs['n_genes'] = n_genes
    adata.var['n_cells'] = n_cells
    return adata




if __name__ == '__main__':
    gene_dataset = CortexDataset()
    adata = dataset2adata(gene_dataset, "type_order")

    print(adata)

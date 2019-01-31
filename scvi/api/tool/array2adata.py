from anndata import AnnData
import numpy as np
import pandas as pd

from scvi.api.tool.get_unique_cell_names import get_unique_cell_names


def array2adata(X, gene_names=None, cell_names=None, cell_types=None,_type='type_order', **kwargs):
    adata = AnnData(X=X.transpose())
    cell_names = cell_names.flatten()
    cell_names = get_unique_cell_names(_type=_type, cell_types=cell_types, labels=cell_names)
    n_genes = pd.Series(np.sum(X, axis=0), index=gene_names)
    n_cells = pd.Series(np.sum(X, axis=1), index=cell_names)
    adata.obs['n_genes'] = n_genes
    adata.var['n_cells'] = n_cells

    return adata

from anndata import AnnData
import numpy as np
import pandas as pd

from scvi.api.tool.get_unique_cell_names import get_unique_cell_names


def array2adata(X, gene_names=None, cell_names=None, cell_types=None,_type='type_order'):
    """
    transform a np.ndarray into an AnnData object
    :param X : :type:np.ndarray input
    :param gene_names: gene names.
    :param cell_names: cell_names/labels of each cells.
    :param cell_types: types of cells.
    :param _type: _type has two choice: 1. "type_order": labels are "cell_types[label] {identifier}"
                                        2. "cell_order": labels are "cell {identifier}"
                                        3. "manual": labels are cell_names

    :return: adata
    """
    adata = AnnData(X=X.transpose())
    cell_names = cell_names.flatten()
    cell_names = get_unique_cell_names(_type=_type, cell_types=cell_types, labels=cell_names)
    print(X)
    n_genes = pd.Series(np.sum(X, axis=0), index=gene_names)
    n_cells = pd.Series(np.sum(X, axis=1), index=cell_names)
    print(n_genes)
    print(n_cells)
    adata.obs['n_genes'] = n_genes
    adata.var['n_cells'] = n_cells

    return adata

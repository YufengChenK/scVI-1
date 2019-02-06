import copy
from typing import Union
import re
import numpy as np
from scvi.dataset import GeneExpressionDataset
import tables
from anndata import AnnData, h5py
import scanpy.api as sc


class AnnGeneDataset(GeneExpressionDataset):
    """
    A class that can transform AnnData into GeneExpressionDataset
    """
    def __init__(self, data: Union[str, AnnData],
                 cell_tag: str = None,
                 gene_tag: str = None,
                 X_tag: str = None,
                 label_tag: str = None,
                 labels: Union[np.ndarray, list] = None):
        """
        constructor of AnnGeneDataset.
        :param data: if data's type is str, then read adata from this path, else it is an anndata object.
        :param cell_tag: the tag of cell in adata.var, default by n_cells and n_counts
        :param gene_tag: the tag of gene in adata.obs, default by n_genes and n_counts
        :param X_tag: the tag of X in adata.uns, default by adata.X
        :param label_tag: the tag of label in adata, deafult by any attrs end with 'label'.
        :param labels: have higher priority then label_tag.
        """
        if isinstance(data, str):  # by default regard it as a filename
            adata = sc.read(data)  # todo: replace with reading .h5ad of scVI
        elif isinstance(data, AnnData):
            adata = data
        else:
            raise TypeError("This Gene Dataset only supports Anndata and string(file name)")
        if X_tag is None:
            X = adata.X
        else:
            d, a = X_tag.split(":")
            X = adata.__dict__["_" + d][a]
        if cell_tag is not None and cell_tag in adata.var:
            cell_types = adata.var[cell_tag].keys().__array__()
        elif "n_cells" in adata.var:
            cell_types = adata.var["n_cells"].keys().__array__()
        elif "n_counts" in adata.var:
            cell_types = adata.var["n_counts"].keys().__array__()
        else:
            raise ValueError()

        if gene_tag is not None and gene_tag in adata.obs:
            genes = adata.obs[gene_tag].keys().__array__()
        elif "n_genes" in adata.obs:
            genes = adata.obs['n_genes'].keys().__array__()
        elif "n_counts" in adata.obs:
            genes = adata.obs['n_counts'].keys().__array__()
        else:
            raise ValueError()

        if labels is not None:
            _labels = labels
        elif label_tag in adata.var:
            _labels = adata.obs[label_tag].__array__()
        else:
            pattern = re.compile(".*labels$")
            for var_item in adata.var:
                if pattern.match(var_item) is not None:
                    _labels = adata.var[var_item].__array__()
                    break
            else:
                if "louvain" in adata.var:
                    _labels = adata.var['louvain'].__array__()
                else:
                    _labels = None
        X = X.transpose()
        if labels is None:
            cell_types = ["Undefined"]
        print(X)
        X, local_mean, local_var, batch_indices, labels = GeneExpressionDataset.get_attributes_from_matrix(X, labels=_labels)
        print("X.shape:", X.shape)

        super(AnnGeneDataset, self).__init__(X,
                                             local_mean,
                                             local_var,
                                             batch_indices,
                                             labels,
                                             gene_names=genes,
                                             cell_types=cell_types,
                                             )

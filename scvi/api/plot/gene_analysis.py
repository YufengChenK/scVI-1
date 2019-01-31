from typing import Union

import seaborn as sns
from anndata import AnnData
import pandas as pd
import matplotlib.pyplot as plt
from scvi.dataset import GeneExpressionDataset
import numpy as np

from scvi.inference import Trainer, Posterior


def different_expression_map(data: Union[GeneExpressionDataset,
                                         AnnData,
                                         np.ndarray,
                                         Trainer],
                             annot=True, cell_numbers=None, nb_gene=None, attr=None, **kwargs):
    if isinstance(data, GeneExpressionDataset):
        X_shape = np.shape(data.X)
        genes = X_shape[1]
        cells = X_shape[0]
        gene_names = data.gene_names
        cell_types = data.cell_types
        if attr is None:
            __X = data.X
        else:
            __X = data.__dict__[attr]

    elif isinstance(data, AnnData):

        cell_attr = kwargs["cell_attr"] if "cell_attr" in kwargs else "n_cells"
        gene_attr = kwargs["gene_attr"] if "gene_attr" in kwargs else "n_genes"

        cell_types = data.var[cell_attr]
        gene_names = data.obs[gene_attr]
        __X = data.X if attr is None else data.uns[attr]
    elif isinstance(data, np.ndarray):
        X_shape = np.shape(data)
        genes = X_shape[1]
        cells = X_shape[0]
        gene_names = ["gene_" + str(i) for i in range(genes)]
        cell_types = ["cell_" + str(i) for i in range(cells)]
        __X = data

    elif isinstance(data, Trainer):
        if "type" in kwargs:
            _type = kwargs['type']
        else:
            _type = "test"

        if _type == "train":
            train:Posterior = data.train_set
            train
            train.differential_expression_table()
        elif _type == "test":
            pass

        else:
            raise ValueError

    _type = kwargs["type"]
    if _type == "cell":
        y_labels = ["cell" + str(i) for i in range(cells)]
        _X = __X
        if cell_numbers is not None:
            df = pd.DataFrame(_X, index=y_labels, columns=gene_names)
        else:
            df = pd.DataFrame(_X[:cell_numbers, :], index=y_labels[:cell_numbers], columns=gene_names)
    elif _type == "cell_type":
        X_ = __X
        y_labels = cell_types
        indices_dict = {cell_type: [] for cell_type in cell_types}
        for index, label in enumerate(data.labels):
            indices_dict[cell_types[label]].append(index)
        X = []
        for y_label in y_labels:
            X.append(np.sum(X_[indices_dict[y_label], :], axis=0))
        _X = np.array(X)
        df = pd.DataFrame(_X, index=y_labels, columns=gene_names)
    else:
        raise ValueError
    if nb_gene is not None and 0 < nb_gene < genes:
        df.truncate(before=gene_names[0], after=gene_names[nb_gene - 1])
    sns.heatmap(df, annot=annot)
    plt.show()

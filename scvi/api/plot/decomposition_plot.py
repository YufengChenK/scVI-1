import math
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from anndata import AnnData
from scvi.dataset import GeneExpressionDataset

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from umap import UMAP


def plot(_type, projection="2d", **kwargs):
    # todo: fix this:different defination of labels in adata and
    print("kwargs:", kwargs)
    if "save_path" in kwargs:
        save_path = kwargs["save_path"]
    else:
        save_path = None
    if "adata" in kwargs or "anndata" in kwargs:
        if "adata" in kwargs:
            adata: AnnData = kwargs["adata"]
        elif "anndata" in kwargs:
            adata: AnnData = kwargs["anndata"]
        else:
            raise TypeError
        X = adata.X
        if "y_name" in kwargs:
            y_name = kwargs["y_name"]
            if y_name in adata.obs:
                y = adata.obs[y_name]
            elif y_name in adata.obsm:
                y = adata.obs[y_name]
            else:
                raise ValueError
        else:
            if "cell_types" in adata.obs:
                y = adata.obs['cell_types']
            if "labels" in adata.obs:
                y = adata.obs['labels']
            if "cell_types" in adata.obsm:
                y = adata.obsm['cell_types']
            if "labels" in adata.obsm:
                y = adata.obsm['labels']
    elif "gene_dataset" in kwargs:

        gene_dataset: GeneExpressionDataset = kwargs['gene_dataset']
        X = gene_dataset.X
        y = gene_dataset.labels
    elif "X" and "labels" in kwargs:
        X = kwargs["X"]
        y = kwargs["labels"]
    elif "X" and "y" in kwargs:
        X = kwargs["X"]
        y = kwargs["y"]

    nb_gene = np.shape(X)[1]
    # shuffled_indices = np.random.permutation(nb_gene)
    # if ("nb_samples" in kwargs and "sample_rate" in kwargs) or \
    #     ("nb_samples" in kwargs and "sample_rate" not in kwargs):
    #     nb_samples = kwargs["nb_samples"]
    #     sample_indices = shuffled_indices[nb_samples]
    #     X = X[sample_indices, :]
    #     y = y[sample_indices]
    # elif "nb_samples" not in kwargs and "sample_rate" in kwargs:
    #     sample_rate = kwargs["sample_rate"]
    #     nb_samples = math.ceil(nb_gene * sample_rate)
    #     sample_indices = shuffled_indices[:nb_samples]
    #     print("sample_indices:", sample_indices.shape)
    #
    #     X = X[sample_indices, :]
    #     y = y[sample_indices, :]


    # todo: reduce duplicate code
    # make it cleaner
    if _type == "t-SNE":
        if projection == "2d":
            tsne = TSNE(n_components=2)
            _X = tsne.fit_transform(X, y)
            plot2d(_X, y, save_path)
        elif projection == "3d":
            tsne = TSNE(n_components=3)
            _X = tsne.fit_transform(X, y)
            plot3d(_X, y, save_path)
    elif _type == "PCA":
        if projection == "2d":
            pca = PCA(n_components=2)
            # print(np.shape(X))
            # print(np.shape(y))
            _X = pca.fit_transform(X, y)
            plot2d(_X, y, save_path)
        elif projection == "3d":
            pca = PCA(n_components=3)
            _X = pca.fit_transform(X, y)
            plot3d(_X, y, save_path)
    elif _type == "umap":
        if projection == "2d":
            umap = UMAP(n_components=2)
            _X = umap.fit_transform(X, y)
            plot2d(_X, y, save_path)
        elif projection == "3d":
            umap = UMAP(n_components=3)
            _X = umap.fit_transform(X, y)
            plot3d(_X, y, save_path)
    else:
        # todo: fill it
        raise TypeError("")


def plot2d(_X, y, save_path=None):
    # print(np.shape(np.shape(y))[0]==2)
    # print(np.shape(_X))
    print(_X)
    if np.shape(np.shape(y))[0] == 2:
        y = y.flatten()
    plt.scatter(_X[:, 0], _X[:, 1], c=y)
    if save_path is not None and isinstance(save_path, str):
        plt.savefig(save_path)
    plt.show()


def plot3d(_X, y, save_path=None):
    if np.shape(np.shape(y))[0] == 2:
        y = y.flatten()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(_X[:, 0], _X[:, 1], _X[:, 2], c=y)
    if save_path is not None and isinstance(save_path, str):
        plt.savefig(save_path)
    plt.show()

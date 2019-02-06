from sklearn.decomposition import PCA

from scvi.api.plot.decomposition_plot import plot


def plot_pca(projection, **kwargs):
    plot("PCA", projection, **kwargs)

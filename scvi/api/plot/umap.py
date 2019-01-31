from scvi.api.plot.decomposition_plot import plot


def plot_umap(projection, **kwargs):
    plot("umap", projection, **kwargs)

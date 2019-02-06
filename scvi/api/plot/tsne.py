
from scvi.api.plot.decomposition_plot import plot


def plot_tsne(projection, **kwargs):
    plot("t-SNE", projection, **kwargs)

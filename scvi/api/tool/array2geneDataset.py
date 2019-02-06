from scvi.dataset import GeneExpressionDataset


class XGeneDataset(GeneExpressionDataset):
    """
    a class that transforms a ndarray into GeneExpressionDataset
    """
    def __init__(self, X, labels=None, **kwargs):
        """
        constructor of XGeneDataset
        :param X: ndarray
        :param labels: labels for X
        :param kwargs:
            allow attribute:
            1. gene_names: gene names for X
            2. cell_types: cell types for X
        """
        nb_genes = X.shape[1]
        nb_cells = X.shape[0]
        if "gene_names" not in kwargs:
            gene_names = ["gene_" + str(geneI) for geneI in range(nb_genes)]
        else:
            gene_names = kwargs['gene_names']
        if "cell_types" not in kwargs:
            cell_types = ["cell_" + str(cellI) for cellI in range(nb_cells)]
        else:
            cell_types = kwargs['cell_types']

        X, local_means, local_vars, batch_indices, labels = GeneExpressionDataset.get_attributes_from_matrix(X, labels=labels)
        super().__init__(X, local_means, local_vars, batch_indices, labels, gene_names, cell_types)



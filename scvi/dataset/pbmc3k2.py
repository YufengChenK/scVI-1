import os
import tarfile
import pandas as pd
import numpy as np
# from scipy import io, warnings
import scipy.io as io
from scipy.sparse import coo_matrix, csr_matrix

from scvi.dataset import Dataset10X
from .dataset import GeneExpressionDataset, arrange_categories


class Pbmc3kDataset(GeneExpressionDataset):
    def __init__(self, save_path="data/", dense=True, remote=True):
        self.save_path = save_path
        self.download_name = "filtered_gene_bc_matrices.tar.gz"
        self.dirname = "filtered_gene_bc_matrices"
        self.url = "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz"
        self.dense = dense
        self.remote = remote

        expression_data, labels_index, gene_names, cell_types, raw_X = self.download_and_preprocess()
        # print(labels_index)
        # self.X = expression_data

        X, local_mean, local_var, batch_indices, labels = GeneExpressionDataset.get_attributes_from_matrix(
            expression_data)

        self.gene_name_index = {gene_name: index for index, gene_name in enumerate(gene_names)}
        # print("self.gene_name_index", self.gene_name_index)
        self._X = np.ascontiguousarray(X, dtype=np.float32)
        self.nb_genes = self.X.shape[1]
        self.local_means = local_mean
        self.local_vars = local_var
        self.batch_indices, self.n_batches = arrange_categories(batch_indices)
        self.labels, self.n_labels = arrange_categories(labels)
        self.x_coord = None
        self.y_coord = None
        self.raw_data = raw_X

    def preprocess(self):
        print("Pbmc3K Dataset preprocess")
        print("Extracting tar file and read data")
        tar = tarfile.open(os.path.join(self.save_path, self.download_name), "r:gz")
        tar.extractall(path=self.save_path)
        tar.close()
        path = (os.path.join(self.save_path, [name for name in os.listdir(self.save_path)
                                              if os.path.isdir(os.path.join(self.save_path, name))][0]) + "/")

        path += os.listdir(path)[0] + '/'
        gene_info = pd.read_csv(path + "genes.tsv", sep="\t", header=None)
        self.gene_names = gene_info.values[:, 1].astype(np.str).ravel()

        if os.path.exists(path + "barcodes.tsv"):
            self.barcodes = pd.read_csv(path + "barcodes.tsv", sep="\t", header=None)
        self.gene_code = gene_info.values[:, 0].astype(np.str).ravel()
        expression_data = io.mmread(path + "matrix.mtx").T
        if self.dense:
            expression_data = expression_data.toarray()
        else:
            expression_data = csr_matrix(expression_data)
        print("end reading data")
        print("start preprocessing it")
        X = np.copy(expression_data)
        min_genes = 200
        min_cells = 3

        number_per_cell = np.sum(X, axis=1)
        cell_subset = number_per_cell >= min_genes
        n_genes = np.sum(X > 0, axis=1)
        number_per_gene = np.sum(X, axis=0)
        gene_subset = number_per_gene >= min_cells
        filter_indices = []
        for i, in_scope in enumerate(gene_subset):
            if in_scope:
                filter_indices.append(i)
        expression_data = expression_data[:, filter_indices]
        self.filtered_gene_index = filter_indices
        mito_genes = [index for index, name in enumerate(self.gene_names[filter_indices]) if name.startswith("MT-")]
        percent_mito = np.sum(expression_data[:, mito_genes], axis=1) / np.sum(expression_data, axis=1)
        n_counts = np.sum(expression_data, axis=1)
        labels_index = []

        for i in range(len(n_genes)):
            if n_genes[i] < 2500 and percent_mito[i] < 0.05:
                labels_index.append(i)
        expression_data = expression_data[labels_index, :]
        self.filtered_label_index = labels_index
        raw_expression_data = np.log1p(expression_data)
        self.raw_expression_data = raw_expression_data
        counts_per_cell_after = 1e4

        # todo: normalize it!

        print("end preprocessing it")
        print("Pbmc3k Dataset preprocessing ends")
        self.expression_data = expression_data
        cell_types = ["undefined"]
        self.raw2filter = {f: index for index, f in enumerate(self.filtered_gene_index)}
        gene_names = self.gene_names
        self.gene_name_index = {gene_name: index for index, gene_name in enumerate(gene_names)}
        return expression_data, labels_index, gene_names, cell_types, X

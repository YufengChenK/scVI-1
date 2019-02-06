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
    def __init__(self, save_path="data/", labels=None, dense=True, remote=True):
        self.save_path = save_path
        self.download_name = "filtered_gene_bc_matrices.tar.gz"
        self.dirname = "filtered_gene_bc_matrices"
        self.url = "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz"
        self.dense = dense
        self.remote = remote

        expression_data, gene_names, raw_X = self.download_and_preprocess()

        if labels is None:
            labels = None
            cell_types = None
        else:
            cell_types = ['CD4 T', 'CD14+ Monocytes',
                          'B', 'CD8 T',
                          'NK', 'FCGR3A+ Monocytes',
                          'Dendritic', 'Megakaryocytes'][:np.max(labels)+1]
            self.cell_index = {cell_type: index for index, cell_type in enumerate(cell_types)}
        self.gene_index = {gene_name: index for index, gene_name in enumerate(gene_names)}
        super(Pbmc3kDataset, self).__init__(
            *GeneExpressionDataset.get_attributes_from_matrix(
                expression_data,
                labels=labels),
            gene_names=np.char.upper(gene_names), cell_types=cell_types)

    def preprocess(self):
        # region extracting
        print("Pbmc3K Dataset preprocess")
        print("Extracting tar file and read data")
        tar = tarfile.open(os.path.join(self.save_path, self.download_name), "r:gz")
        tar.extractall(path=self.save_path)
        tar.close()
        print([name for name in os.listdir(self.save_path) if os.path.isdir(os.path.join(self.save_path, name))])
        path = (os.path.join(self.save_path, [name for name in os.listdir(self.save_path)
                                              if os.path.isdir(os.path.join(self.save_path, name)) and
                                              self.dirname == name][0]) + "/")
        print("path:", path)
        path += os.listdir(path)[0] + '/'
        gene_info = pd.read_csv(path + "genes.tsv", sep="\t", header=None)
        gene_names = gene_info.values[:, 1].astype(np.str).ravel()
        if os.path.exists(path + "barcodes.tsv"):
            self.barcodes = pd.read_csv(path + "barcodes.tsv", sep="\t", header=None)
        self.gene_code = gene_info.values[:, 0].astype(np.str).ravel()
        expression_data = io.mmread(path + "matrix.mtx").T
        if self.dense:
            expression_data = expression_data.toarray()
        else:
            expression_data = csr_matrix(expression_data)
        print("end reading data")
        # endregion
        print("start preprocessing it")
        raw_X = np.copy(expression_data)
        min_genes = 200
        min_cells = 3

        number_per_cell = np.sum(raw_X, axis=1)
        cell_subset = number_per_cell >= min_genes
        filter_indices = []
        for i, in_scope in enumerate(cell_subset):
            if in_scope:
                filter_indices.append(i)
        X = raw_X[filter_indices, :]
        n_genes = np.sum(raw_X > 0, axis=1)
        number_per_gene = np.sum(raw_X, axis=0)
        gene_subset = number_per_gene >= min_cells
        filter_indices = []
        for i, in_scope in enumerate(gene_subset):
            if in_scope:
                filter_indices.append(i)
        X = X[:, filter_indices]

        filter = filter_indices.copy()
        mito_genes = [index for index, name in enumerate(gene_names[filter_indices]) if name.startswith("MT-")]

        percent_mito = np.sum(X[:, mito_genes], axis=1) / np.sum(X, axis=1)
        n_counts = np.sum(X, axis=1)
        labels_index = []
        for i in range(len(n_genes)):
            if n_genes[i] < 2500 and percent_mito[i] < 0.05:
                labels_index.append(i)
        X = X[labels_index, :]
        self.filtered_label_index = labels_index
        min_genes = 1
        counts_per_cell_after = 1e4
        counts_per_cell = np.sum(X, axis=1)
        cell_subset = counts_per_cell >= min_genes
        X = X[cell_subset]
        counts_per_cell = counts_per_cell[cell_subset]
        counts_per_cell += counts_per_cell == 0
        counts_per_cell /= counts_per_cell_after
        X /= counts_per_cell[:, np.newaxis]
        X = np.log1p(X)
        X = np.expm1(X)
        # highly variable genes
        mean = X.mean(axis=0)
        mean_sq = np.multiply(X, X).mean(axis=0)
        var = (mean_sq - mean ** 2) * (X.shape[0] / (X.shape[0] - 1))
        mean[mean == 0] = 1e-12
        min_mean = 0.0125
        max_mean = 3
        n_bins = 20
        dispersion = var / mean
        dispersion[dispersion == 0] = np.nan
        dispersion = np.log(dispersion)
        mean = np.log1p(mean)
        mean_bin = pd.cut(mean, bins=n_bins)
        df = pd.DataFrame()
        df['mean'] = mean
        df['dispersion'] = dispersion
        df['mean_bin'] = mean_bin
        # print(df)
        disp_grouped = df.groupby('mean_bin')['dispersion']
        # print(df.groupby('mean_bin').count())
        disp_mean_bin = disp_grouped.mean()
        disp_std_bin = disp_grouped.std(ddof=1)
        one_gene_per_bin = disp_std_bin.isnull()

        disp_std_bin[one_gene_per_bin] = disp_mean_bin[one_gene_per_bin].values

        dispersion_norm = ((df['dispersion'].values  # use values here as index differs
                            - disp_mean_bin[df['mean_bin']].values) / disp_std_bin[df['mean_bin']].values).astype(
            "float32")

        disp_mean_bin[one_gene_per_bin] = 0

        min_disp = 0.5
        dispersion_norm[np.isnan(dispersion_norm)] = 0  # similar to Seurat
        highly_variable = np.logical_and.reduce([mean > min_mean, mean < max_mean, dispersion_norm > min_disp])
        X = X[:, highly_variable]
        gene_clusters = ['CD4 T', 'CD14+ Monocytes',
                         'B', 'CD8 T',
                         'NK', 'FCGR3A+ Monocytes',
                         'Dendritic', 'Megakaryocytes']
        gene_name_list = []
        for index, b in enumerate(highly_variable):
            if b:
                # print(filter[index])
                gene_name_list.append(gene_names[filter[index]])

        return X, gene_name_list, raw_X

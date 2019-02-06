import numpy as np
import scanpy.api as sc

sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.set_figure_params(dpi=80)  # low dpi (dots per inch) yields small inline figures
sc.logging.print_version_and_date()
sc.logging.print_versions_dependencies_numerics()
results_file = './write/pmbc3k.h5ad'
path = './data10X/pbmc3k/filtered_gene_bc_matrices/hg19/'
adata = sc.read(path + 'matrix.mtx', cache=True).transpose()
adata.var_names = np.genfromtxt(path + 'genes.tsv', dtype=str)[:, 1]
adata.obs_names = np.genfromtxt(path + 'barcodes.tsv', dtype=str)
adata.var_names_make_unique()
print(adata)

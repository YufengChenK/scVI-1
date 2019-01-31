from typing import Union

import numpy as np
import torch
from anndata import AnnData

from scvi.api.tool.array2geneDataset import XGeneDataset
from scvi.dataset import GeneExpressionDataset
from scvi.inference import Posterior
from scvi.inference.inference import UnsupervisedTrainer, AdapterTrainer
from scvi.models.vae import VAE as _VAE


class Model:
    def __init__(self, train_size=0.8, use_cuda=True, frequency=5, n_epoch=400, lr=1e-3):
        self.train_size = train_size
        self.use_cuda = use_cuda
        self.frequency = frequency
        self.n_epoch = n_epoch
        self.lr = lr
        self.trained = False
        self.trainer = None
        self.model = None

    def imputation(self, gene_dataset):
        assert self.trained
        posterior = Posterior(self.model, gene_dataset, use_cuda=self.use_cuda,
                              data_loader_kwargs={"batch_size": 128, "pin_memory":
                                  self.use_cuda})
        return posterior.imputation()

    def get_training_imputation(self):
        assert self.trained
        return self.trainer.get_all_latent_and_imputed_values()["imputed_values"]

    def differential_expression(self, gene_dataset: GeneExpressionDataset):
        assert self.trained
        posterior = Posterior(self.model, gene_dataset, use_cuda=self.use_cuda, data_loader_kwargs={"batch_size": 128,
                                                                                                    "pin_memory":
                                                                                                        self.use_cuda})
        # todo: change default setting
        return posterior.differential_expression_score(cell_type=gene_dataset.cell_types, genes=gene_dataset.gene_names)

    def classify(self, X):
        raise NotImplementedError()

    def transform(self, X):
        raise NotImplementedError()

    def fit(self, data: Union[AnnData, GeneExpressionDataset, np.ndarray], _type, **kwargs):
        raise NotImplementedError()

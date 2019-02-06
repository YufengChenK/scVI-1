import numpy as np
import torch
from anndata import AnnData

from scvi.api.model.model import Model
from scvi.api.tool.array2geneDataset import XGeneDataset
from scvi.dataset import GeneExpressionDataset
from scvi.inference import Posterior
from scvi.inference.inference import UnsupervisedTrainer, AdapterTrainer
from scvi.models.vaec import VAEC as _VAEC


class VAEC(Model):
    def __init__(self, n_input: int, n_batch: int = 0, n_labels: int = 0, n_hidden: int = 128, n_latent: int = 10,
                 n_layers: int = 1, dropout_rate: float = 0.1, y_prior=None, dispersion: str = "gene",
                 log_variational: bool = True,
                 reconstruction_loss: str = "zinb", train_size=0.8, use_cuda=True, frequency=5, n_epoch=400, lr=1e-3):
        super().__init__(train_size, use_cuda, frequency, n_epoch, lr)
        self.model = _VAEC(n_input, n_batch, n_labels,
                           n_hidden, n_latent, n_layers,
                           dropout_rate, y_prior, dispersion, log_variational, reconstruction_loss)

    def fit_genedataset(self, gene_dataset: GeneExpressionDataset, _type, **kwargs):
        if _type == "unsupervised":
            self.trainer = UnsupervisedTrainer(self.model, gene_dataset, train_size=self.train_size,
                                               use_cuda=self.use_cuda,
                                               frequency=self.frequency)
            self.trainer.train(n_epochs=self.n_epoch, lr=self.lr)

            return self.trainer
        if _type == "adapter":
            if "label_class" in kwargs:
                posterior_test = min(kwargs["posterior_test"])
            else:
                raise ValueError("You shouldn't use adapter trainer")

            self.trainer = AdapterTrainer(self.model, gene_dataset, posterior_test=posterior_test,
                                          frequency=self.frequency)
            self.trainer.train(n_epochs=self.n_epoch, lr=self.lr)
            return self.trainer
        else:
            raise ValueError("it should unsupervised or 'semi-supervised'")

    def fit(self, data, _type, **kwargs):
        if isinstance(data, AnnData):
            from scvi.api.tool.annDataset import AnnGeneDataset
            gene_dataset = AnnGeneDataset(data)
            self.gene_dataset = gene_dataset
            self.fit_genedataset(gene_dataset, _type, **kwargs)
        elif isinstance(data, GeneExpressionDataset):
            gene_dataset = data
            self.fit_genedataset(gene_dataset, _type, **kwargs)
        elif isinstance(data, np.ndarray):
            if "label" in kwargs:
                gene_dataset = XGeneDataset(data, labels=kwargs['labels'])
            else:
                gene_dataset = XGeneDataset(data)
            self.fit_genedataset(gene_dataset, _type=_type, **kwargs)
        else:
            raise TypeError("Cannot support this type, please input\n"
                            "1. AnnData\n"
                            "2. GeneExpressionDataset\n"
                            "3. ndarray")
        self.trained = True

    def classify(self, X):
        return self.model.classify(X)

    def get_train_posterior(self):
        assert self.trained
        return self.trainer.train_set

    def get_test_posterior(self):
        assert self.trained
        return self.trainer.test_set

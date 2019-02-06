import json
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition.pca import PCA
from scvi.dataset import Pbmc3kDataset
from scvi.models import *
from scvi.inference import UnsupervisedTrainer
import os

with open('tests/notebooks/basic_tutorial.config.json') as f:
    config = json.load(f)
print(config)

gene_dataset = Pbmc3kDataset()

n_epochs = 400
lr = 1e-3
use_batches = False
use_cuda = False

# dataset.expression_data[:, dataset]
print(np.shape(gene_dataset.X))
vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * use_batches)
trainer = UnsupervisedTrainer(vae, gene_dataset, train_size=0.75, use_cuda=use_cuda, frequency=5)
trainer.train(n_epochs=n_epochs, lr=lr)

print(trainer.train_set.get_latent())

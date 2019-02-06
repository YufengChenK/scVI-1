from scvi.dataset import PbmcDataset
from scanpy.api.datasets import pbmc68k_reduced
from scvi.api.tool.annDataset import AnnGeneDataset
from scvi.api.model.vae import VAE

pbmc = pbmc68k_reduced()
# print(pbmc)
gene_dataset = AnnGeneDataset(pbmc, None, None, None, None, None)

vae = VAE(gene_dataset.X.shape[1])
vae.fit(gene_dataset)
print(vae.transform(gene_dataset.X))

from scvi.inference import UnsupervisedTrainer
from scvi.models.vae import VAE
vae2 = VAE(gene_dataset.nb_genes)
trainer = UnsupervisedTrainer(vae2, gene_dataset)
trainer.train()
print(trainer.train_set.get_latent())


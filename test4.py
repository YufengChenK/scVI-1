from scanpy.datasets import pbmc68k_reduced
from scvi.inference import UnsupervisedTrainer
from scvi.models import VAE
from scvi.api import AnnGeneDataset


pbmc = pbmc68k_reduced()
gene_dataset = AnnGeneDataset(pbmc)
vae = VAE(gene_dataset.nb_genes)
trainer = UnsupervisedTrainer(vae, gene_dataset)
trainer.train(400)
test = trainer.test_set
print(test.get_latent())

# In[1]:
#test vae
from scvi.api.model.vae import VAE
from scvi.dataset.cortex import CortexDataset
gene_dataset = CortexDataset()
vae = VAE(gene_dataset.nb_genes)

# In[2]

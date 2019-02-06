#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scvi.api.dataset import CortexDataset
gene_dataset = CortexDataset()


# In[2]:


from scvi.inference import UnsupervisedTrainer
from scvi.models.vae import VAE
vae = VAE(gene_dataset.nb_genes)
trainer = UnsupervisedTrainer(vae, gene_dataset)
trainer.train(1)


# In[3]:


test_posterior = trainer.test_set


# In[4]:


from scvi.api import posterior2adata
adata = posterior2adata(test_posterior)


# In[5]:


import scanpy.api as sc
sc.pl.highest_expr_genes(adata)


# In[6]:


from scanpy.plotting.tools.scatterplots import plot_scatter

plot_scatter(adata, basis="scVI")


# In[7]:


from sklearn.cluster import KMeans
kmeans = KMeans()
kmeans.fit_transform(adata.obsm["X_scVI"])

adata.obs['kmeans'] = kmeans.labels_
plot_scatter(adata, basis="scVI", color='kmeans')


# In[11]:


print(adata.var['n_genes'])


# In[ ]:





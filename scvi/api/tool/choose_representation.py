"""
A more general version of `choose representation`
"""
from scanpy.preprocessing.simple import N_PCS, pca
import scanpy.logging as logg
from scanpy.tools import tsne
from scvi.api.model.model import Model
from scvi.api.model.vae import VAE


def choose_representation(adata, use_rep=None, n_pcs=None, **kwarg):
    """
    an general version of choose_representation in scanpy.
    Choose what to present.
    add more dimension reduction algorithm ("PCA", "TSNE", "SCVI")

    :param adata:
    :param use_rep: the attribute in adata.obsm, use what to represent
    :param n_pcs: ignored is use_rep is not None. how many principle components it has.
    :param kwarg:
            attributes:
            1. 'tag':
            2. 'pre_method': 4 values legal: "PCA"/"pca", "tsne"/"t-SNE"/"t-sne", "scvi", default by "PCA"
            3. '_type': _type in scvi.api.model.vae.VAE. 2 values legal: "unsupervised", "adapter"
    :return:
    """
    if use_rep is None and n_pcs == 0:  # backwards compat for specifying `.X`
        use_rep = 'X'
    if use_rep is None:
        if adata.n_vars > N_PCS:
            if "tag" in kwarg and kwarg['tag'] in adata.obsm.keys():
                tag = kwarg['tag']
            else:
                tag = "X_pca"
            if tag in adata.obsm.keys():
                if n_pcs is not None and n_pcs > adata.obsm[tag].shape[1]:
                    raise ValueError(
                        '`{}` does not have enough PCs.'.format(tag))
                X = adata.obsm[tag][:, :n_pcs]
                logg.info('    using \'{}\' with n_pcs = {}'
                          .format(tag, X.shape[1]))
                return X
            elif "pre_method" in kwarg:
                if type(kwarg['pre_method']) == "str":
                    pre_method = kwarg['pre_method']
                    if pre_method == "PCA" or pre_method == "pca":
                        X = pca(adata.X)
                        adata.obsm['X_pca'] = X[:, :n_pcs]
                        return X

                    elif pre_method == "tsne" or pre_method == "t-SNE" or pre_method == "t-sne":
                        X = tsne(adata.X, n_pcs=N_PCS, use_rep=use_rep, **kwarg)
                        adata.obsm["X_tsne"] = X[:, n_pcs]
                        return X

                    elif pre_method.lower() == "scvi":
                        if "model" in kwarg:
                            model:Model = kwarg["model"]
                            if model.trained:
                                X = model.transform(adata.X)  # or get_latent
                            else:
                                _type = kwarg["type"] if "type" in kwarg else "unsupervised"
                                X = model.fit(adata.X, _type=_type, **kwarg)
                                X = model.transform(X)
                            adata.obsm['X_scvi'] = X[:, n_pcs]
                            return X
                        else:
                            X = adata.X
                            n_input = X.shape[1]
                            vae = VAE(n_input, **kwarg)
                            _type = kwarg["type"] if "type" in kwarg else "unsupervised"
                            X = vae.fit(X, _type=_type, **kwarg)
                            X = vae.transform(X)
                            adata.obsm['X_scvi'] = X[:, n_pcs]
                            return X
            else:
                "PCA"
                X = pca(adata.X)
                adata.obsm['X_pca'] = X[:, :n_pcs]
                return X

        else:
            logg.info('    using data matrix X directly')
            return adata.X
    else:
        if use_rep in adata.obsm.keys():
            return adata.obsm[use_rep]
        elif use_rep == 'X':
            return adata.X
        else:
            raise ValueError(
                'Did not find {} in `.obsm.keys()`. '
                'You need to compute it first.'.format(use_rep))

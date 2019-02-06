from scvi.api.model.model import Model
from scvi.models import SCANVI


class ScanVI(Model):
    def __init__(self, n_input, n_batch, n_labels, n_hidden, n_latent, n_layers, dropout_rate, dispersion,
                 log_variational, reconstruction_loss, y_prior, labels_groups, use_labels_groups, classifier_parameters,
                 train_size=0.8, use_cuda=True, frequency=5, n_epoch=400, lr=1e-3):
        super().__init__(train_size, use_cuda, frequency, n_epoch, lr)
        self.model = SCANVI(n_input, n_batch, n_labels, n_hidden, n_latent, n_layers, dropout_rate, dispersion,
                            log_variational, reconstruction_loss, y_prior, labels_groups, use_labels_groups,
                            classifier_parameters)

    def classify(self, X):
        self.model.classify(X)

    def get_latent(self, X, y=None):
        self.model.get_latents(X, y)

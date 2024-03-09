# Author: Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 14/06/2023

# Packages to import
import torch
import numpy as np

from .vae_utils import EarlyStopper, check_nan_inf, sample_from_dist
from .vae_modules import LatentSpaceGaussian, Encoder, Decoder, LogLikelihoodLoss


class VariationalAutoencoder(torch.nn.Module):
    def __init__(self, params):
        super(VariationalAutoencoder, self).__init__()

        # Hyperparameters
        self.input_dim = params['input_dim']
        self.feat_distributions = params['feat_distributions']
        assert len(self.feat_distributions) == self.input_dim
        self.latent_dim = params['latent_dim']
        self.hidden_size = params['hidden_size']
        self.early_stop = params['early_stop']

        # Standard VAE modules. Encoder and Latent Space
        self.latent_space = LatentSpaceGaussian(self.latent_dim)
        self.Encoder = Encoder(input_dim=self.input_dim, hidden_dim=self.hidden_size,
                               output_dim=self.latent_space.latent_params)
        self.Decoder = Decoder(latent_dim=self.latent_dim, hidden_size=self.hidden_size,
                               feat_dists=self.feat_distributions)

        # Define losses
        self.rec_loss = LogLikelihoodLoss(self.feat_distributions)

        if self.early_stop:
            self.early_stopper = EarlyStopper(patience=30, min_delta=2)

    def forward(self, input_data):
        latent_output = self.Encoder(input_data)
        latent_params = self.latent_space.get_latent_params(latent_output)
        z = self.latent_space.sample_latent(latent_params)  # Sample the latent space using the reparameterization trick
        check_nan_inf(z, 'Latent space')
        cov_params = self.Decoder(z)
        check_nan_inf(cov_params, 'Decoder')
        return {'z': z, 'cov_params': cov_params, 'latent_params': latent_params}

    def predict(self, x, device=torch.device('cpu')):
        cov = np.array(x)
        self.eval()
        out = self(torch.from_numpy(cov).to(device).float())
        cov_params = out['cov_params'].detach().cpu().numpy()
        out['cov_samples'] = sample_from_dist(cov_params, self.feat_distributions)
        return out

    def fit_epoch(self, data, optimizer, batch_size=64, device=torch.device('cpu')):
        epoch_results = {'loss_tr': 0.0, 'loss_va': 0.0, 'kl_tr': 0.0, 'kl_va': 0.0, 'll_cov_tr': 0.0, 'll_cov_va': 0.0}
        cov_train, mask_train, cov_val, mask_val = data

        # Train epoch
        cov_val = torch.from_numpy(cov_val).to(device).float()
        mask_val = torch.from_numpy(mask_val).to(device).float()
        n_batches = int(np.ceil(cov_train.shape[0] / batch_size).item())
        for batch in range(n_batches):
            # Get (X, y) of the current mini batch/chunk
            index_init = batch * batch_size
            index_end = min(((batch + 1) * batch_size, cov_train.shape[
                0]))  # Use the min to prevent errors due to samples being smaller than batch_size
            cov_train_batch = cov_train[index_init: index_end]
            mask_train_batch = mask_train[index_init: index_end]

            self.train()
            cov_train_batch = torch.from_numpy(cov_train_batch).to(device).float()
            mask_train_batch = torch.from_numpy(mask_train_batch).to(device).float()

            # Generate output params
            out = self(cov_train_batch)
            latent_params = out['latent_params']
            cov_params = out['cov_params']

            # Compute losses
            optimizer.zero_grad()
            loss_kl = self.latent_space.kl_loss(latent_params)
            loss_cov = self.rec_loss(cov_params, cov_train_batch, mask_train_batch)
            loss = loss_kl + loss_cov
            loss.backward()
            optimizer.step()

            # Save data
            epoch_results['loss_tr'] += loss.item()
            epoch_results['kl_tr'] += loss_kl.item()
            epoch_results['ll_cov_tr'] += loss_cov.item()

            # Validation step
            self.eval()
            with torch.no_grad():
                out = self(cov_val)
                latent_params = out['latent_params']
                cov_params = out['cov_params']
                loss_kl = self.latent_space.kl_loss(latent_params)
                loss_cov = self.rec_loss(cov_params, cov_val, mask_val)
                loss = loss_kl + loss_cov

                # Save data
                epoch_results['loss_va'] += loss.item()
                epoch_results['kl_va'] += loss_kl.item()
                epoch_results['ll_cov_va'] += loss_cov.item()

        if self.early_stop:
            self.early_stopper.early_stop(epoch_results['loss_va'])

        return epoch_results

    def fit(self, data, train_params):
        training_stats = {'loss_tr': [], 'loss_va': [], 'kl_tr': [], 'kl_va': [], 'll_cov_tr': [], 'll_cov_va': []}

        optimizer = torch.optim.Adam(self.parameters(), lr=train_params['lr'])

        for epoch in range(train_params['n_epochs']):
            # Configure input data and missing data mask
            x_train, mask_train, x_val, mask_val = data
            assert mask_train.shape == x_train.shape
            assert mask_val.shape == x_val.shape
            ep_data = np.array(x_train), np.array(mask_train), np.array(x_val), np.array(mask_val)

            epoch_results = self.fit_epoch(ep_data, optimizer, batch_size=train_params['batch_size'],
                                           device=train_params['device'])

            for key in epoch_results.keys():
                training_stats[key].append(epoch_results[key])

            if epoch % 1 == 0:
                print('Iteration = ', epoch,
                      '; train loss = ', '{:.2f}'.format(epoch_results['loss_tr']),
                      '; val loss = ', '{:.2f}'.format(epoch_results['loss_va']),
                      '; ll_cov_tr = ', '{:.2f}'.format(np.sum(epoch_results['ll_cov_tr'])),
                      '; ll_cov_va = ', '{:.2f}'.format(np.sum(epoch_results['ll_cov_va'])),
                      '; kl_tr = ', '{:.2f}'.format(epoch_results['kl_tr']),
                      '; kl_va = ', '{:.2f}'.format(epoch_results['kl_va']))

            if self.early_stop and self.early_stopper.stop:
                print('EARLY STOP')
                break

        return training_stats

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

# Author: Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 12/09/2023


# Packages to import
import torch

import numpy as np

from sklearn.mixture import BayesianGaussianMixture
from base_model.vae_model import VariationalAutoencoder
from base_model.vae_utils import check_nan_inf, sample_from_dist


class Generator(VariationalAutoencoder):
    """
    Module implementing Synthethic data generator
    """

    def __init__(self, params):
        # Initialize Generator parameters and modules
        super(Generator, self).__init__(params)
        self.bgm = None
        self.gen_info = None
        self.gauss_gen_info = None
        self.bgm_posterior_samples = int(params.get('bgm_posterior_samples', 0))
        self.bgm_components = int(params.get('bgm_components', self.latent_dim))
        self.bgm_n_init = int(params.get('bgm_n_init', 10))
        self.bgm_max_iter = int(params.get('bgm_max_iter', 5000))
        self.bgm_max_try = int(params.get('bgm_max_try', 100))
        if self.bgm_posterior_samples < 0 or min(self.bgm_components, self.bgm_n_init,
                                                 self.bgm_max_iter, self.bgm_max_try) < 1:
            raise ValueError('Invalid BGM sampling or fitting parameters')

    def train_latent_generator(self, x):  # Train latent generator using x data as input
        if self.bgm is not None:
            raise RuntimeWarning('[WARNING] BGM is being retrained')

        vae_data = self.predict(x)
        mu_latent_param, log_var_latent_param = vae_data['latent_params']
        fit_latent = mu_latent_param
        if self.bgm_posterior_samples:
            eps = torch.randn((self.bgm_posterior_samples,) + tuple(mu_latent_param.shape),
                              device=mu_latent_param.device)
            fit_latent = (mu_latent_param.unsqueeze(0) +
                          torch.exp(0.5 * log_var_latent_param).unsqueeze(0) * eps).reshape(-1, self.latent_dim)

        # Fit GMM to the mean
        converged = False
        bgm = None
        n_try = 0
        max_try = self.bgm_max_try
        # NOTE: this code is for Gaussian latent space, change it if using a different one!
        while not converged and n_try < max_try:  # BGM may not converge: try different times until it converges
            # (or it reaches a max number of iterations)
            n_try += 1
            bgm = BayesianGaussianMixture(n_components=min(self.bgm_components, len(fit_latent)),
                                          random_state=42 + n_try, reg_covar=1e-5,
                                          n_init=self.bgm_n_init, max_iter=self.bgm_max_iter).fit(
                fit_latent.detach().cpu().numpy())
            converged = bgm.converged_

        if not converged:
            raise RuntimeError('BGM did not converge after ' + str(n_try) + ' attempts')
        else:
            self.bgm = {'bgm': bgm,
                        'log_var_mean': np.mean(log_var_latent_param.detach().cpu().numpy(),
                                                axis=0)}  # BGM data to generate patients

    def sample_space(self, z, mu_sample, log_var_sample, model_type='bgm'):
        cov_params = self.Decoder(z)
        check_nan_inf(cov_params, 'Decoder')
        cov_params = cov_params.detach().cpu().numpy()
        cov_samples = sample_from_dist(cov_params, self.feat_distributions)
        if model_type == 'bgm':
            if self.gen_info is None:
                self.gen_info = {'z': z.detach().cpu().numpy(), 'cov_params': cov_params, 'cov_samples': cov_samples,
                                 'latent_params': [mu_sample, log_var_sample]}
            else:
                # Concatenate new samples
                self.gen_info['z'] = np.concatenate((self.gen_info['z'], z.detach().cpu().numpy()), axis=0)
                self.gen_info['cov_params'] = np.concatenate((self.gen_info['cov_params'], cov_params), axis=0)
                self.gen_info['cov_samples'] = np.concatenate((self.gen_info['cov_samples'], cov_samples), axis=0)
                self.gen_info['latent_params'][0] = np.concatenate((self.gen_info['latent_params'][0], mu_sample),
                                                                   axis=0)
                self.gen_info['latent_params'][1] = np.concatenate((self.gen_info['latent_params'][1], log_var_sample),
                                                                   axis=0)
        elif model_type == 'gauss':
            if self.gauss_gen_info is None:
                self.gauss_gen_info = {'z': z.detach().cpu().numpy(), 'cov_params': cov_params,
                                       'cov_samples': cov_samples, 'latent_params': [mu_sample, log_var_sample]}
            else:
                # Concatenate new samples
                self.gauss_gen_info['z'] = np.concatenate((self.gauss_gen_info['z'], z.detach().cpu().numpy()), axis=0)
                self.gauss_gen_info['cov_params'] = np.concatenate((self.gauss_gen_info['cov_params'], cov_params),
                                                                   axis=0)
                self.gauss_gen_info['cov_samples'] = np.concatenate((self.gauss_gen_info['cov_samples'], cov_samples),
                                                                    axis=0)
                self.gauss_gen_info['latent_params'][0] = np.concatenate(
                    (self.gauss_gen_info['latent_params'][0], mu_sample), axis=0)
                self.gauss_gen_info['latent_params'][1] = np.concatenate(
                    (self.gauss_gen_info['latent_params'][1], log_var_sample), axis=0)

    def generate(self, n_gen=1000):
        if self.bgm is None:
            print('[WARNING] BGM  is not trained, try calling train_latent_generator before calling generate')
        else:
            if self.bgm_posterior_samples:
                self.bgm['bgm'].random_state = np.random.RandomState(np.random.randint(0, 2**31 - 1))
            mu_sample = self.bgm['bgm'].sample(n_gen)[0]
            log_var_sample = np.tile(self.bgm['log_var_mean'], (n_gen, 1))

            if self.bgm_posterior_samples:
                z = torch.from_numpy(mu_sample).float()
            else:
                z = self.latent_space.sample_latent([torch.from_numpy(mu_sample).float(),
                                                     torch.from_numpy(log_var_sample).float()])

            check_nan_inf(z, 'GMM latent space')
            self.sample_space(z, mu_sample, log_var_sample)

    def generate_gauss(self, n_gen=1000):
        mu_sample = np.zeros((n_gen, self.latent_dim))
        log_var_sample = np.zeros((n_gen, self.latent_dim))
        z = self.latent_space.sample_latent([torch.from_numpy(mu_sample).float(),
                                             torch.from_numpy(log_var_sample).float()])
        check_nan_inf(z, 'Gauss latent space')
        self.sample_space(z, mu_sample, log_var_sample, 'gauss')

    def validate_samples_range(self, data_manager, n_gen, gauss=False, max_iterations=100):
        if gauss:
            # Denormalize generated samples to check data ranges
            self.gauss_gen_info = data_manager.postprocess_gen_data(self.gauss_gen_info)

            # If number of gen samples is less than n_gen, repeat the process
            for _ in range(max_iterations):
                if self.gauss_gen_info['cov_samples'].shape[0] >= n_gen:
                    break
                self.generate_gauss(n_gen=n_gen - self.gauss_gen_info['cov_samples'].shape[0])
                self.gauss_gen_info = data_manager.postprocess_gen_data(self.gauss_gen_info)
            if self.gauss_gen_info['cov_samples'].shape[0] < n_gen:
                raise RuntimeError('Unable to generate enough valid Gaussian samples')
        else:
            # Denormalize generated samples to check data ranges
            self.gen_info = data_manager.postprocess_gen_data(self.gen_info)

            # If number of gen samples is less than n_gen, repeat the process
            for _ in range(max_iterations):
                if self.gen_info['cov_samples'].shape[0] >= n_gen:
                    break
                self.generate(n_gen=n_gen - self.gen_info['cov_samples'].shape[0])
                self.gen_info = data_manager.postprocess_gen_data(self.gen_info)
            if self.gen_info['cov_samples'].shape[0] < n_gen:
                raise RuntimeError('Unable to generate enough valid BGM samples')

    def compare_latent_spaces(self, data, path, dataset_name=''):
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        # Obtain all latent spaces
        vae_info = self.predict(data)
        z = vae_info['z'].detach().numpy()
        z_bgm = self.gen_info['z']
        z_gauss = self.gauss_gen_info['z']

        # Take just 2000 samples to plot
        if z.shape[0] > 500:
            # Take 500 samples from each
            import pandas as pd
            z_pd = pd.DataFrame(z)
            z = np.array(z_pd.sample(n=500, random_state=0).reset_index(drop=True))
            z_bgm_pd = pd.DataFrame(z)
            z_bgm = np.array(z_bgm_pd.sample(n=500, random_state=0).reset_index(drop=True))
            z_gauss_pd = pd.DataFrame(z_gauss)
            z_gauss = np.array(z_gauss_pd.sample(n=500, random_state=0).reset_index(drop=True))


        # Plot TSNE representation of latent spaces
        tsne = TSNE(n_components=2, random_state=0)
        z_concat = np.concatenate([z, z_bgm, z_gauss])
        z_tsne = tsne.fit_transform(z_concat)
        plt.figure(figsize=(8, 6))
        plt.scatter(z_tsne[:z.shape[0], 0], z_tsne[:z.shape[0], 1], color='blue', alpha=0.5, s=60)
        plt.scatter(z_tsne[(z.shape[0] + z_bgm.shape[0]):, 0], z_tsne[(z.shape[0] + z_bgm.shape[0]):, 1], alpha=0.2,
                color='green', s=20)
        plt.scatter(z_tsne[z.shape[0]:(z.shape[0] + z_bgm.shape[0]), 0],
                z_tsne[z.shape[0]:(z.shape[0] + z_bgm.shape[0]), 1], s=7, color='orange')
        plt.legend(['Real', 'VAE-Gauss', 'VAE-BGM'], loc='lower left', framealpha=1.0, fontsize=16)
        plt.xticks(rotation=0, fontsize=14)
        plt.yticks(fontsize=14)
        plt.title(dataset_name, fontsize=18)
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(path + 'gauss_bgm_vae_tsne.png')
        plt.close()


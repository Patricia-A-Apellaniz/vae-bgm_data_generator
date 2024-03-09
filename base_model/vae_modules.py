# Author: Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 14/06/2023

# Import libraries
import torch
import numpy as np

from torch import nn
from torch.nn import functional as F
from .vae_utils import get_dim_from_type, get_activations_from_types, check_nan_inf


class LatentSpaceGaussian(object):
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        self.latent_params = 2 * latent_dim  # Two parameters are needed for each Gaussian distribution

    def get_latent_params(self, x):
        x = x.view(-1, 2, self.latent_dim)
        mu = x[:, 0, :]
        log_var = x[:, 1, :]
        return mu, log_var

    def sample_latent(self, latent_params):
        mu, log_var = latent_params
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        z = mu + (eps * std)  # sampling as if coming from the input space
        return z

    def kl_loss(self, latent_params):
        mu, log_var = latent_params
        kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())  # Kullback-Leibler divergence
        return kl / mu.shape[0]


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, grad_clip=1000.0, latent_limit=10.0):
        super(Encoder, self).__init__()
        self.enc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.enc2 = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        # Note that this is a Gaussian encoder: latent_dim is the number of gaussian
        # distributions (hence, we need 2 * latent_dim parameters!)
        self.grad_clip = grad_clip
        self.latent_limit = latent_limit  # To limit the latent space values

    def forward(self, inp):
        x = self.enc1(inp)
        if x.requires_grad:
            x.register_hook(lambda x: x.clamp(min=-self.grad_clip, max=self.grad_clip))
        x = F.relu(x)
        x = self.enc2(x)
        if x.requires_grad:
            x.register_hook(lambda x: x.clamp(min=-self.grad_clip, max=self.grad_clip))
        x = torch.tanh(x) * self.latent_limit
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim, feat_dists, max_k=10000.0, dropout_p=0.2, hidden_size=50):
        super(Decoder, self).__init__()
        self.feat_dists = feat_dists
        self.out_dim = get_dim_from_type(self.feat_dists)
        self.dec1 = nn.Linear(in_features=latent_dim, out_features=hidden_size)
        self.dec2 = nn.Linear(in_features=hidden_size, out_features=self.out_dim)
        self.max_k = max_k
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, z):
        x = F.relu(self.dec1(z))
        x = self.dropout(x)
        x = self.dec2(x)
        x = get_activations_from_types(x, self.feat_dists, max_k=self.max_k)
        return x


class LogLikelihoodLoss(nn.Module):
    def __init__(self, feat_dists):
        super(LogLikelihoodLoss, self).__init__()
        self.feat_dists = feat_dists

    def forward(self, inputs, targets, imp_mask):
        index_x = 0  # Index of parameters (do not confound with index_type below, which is the index of the
        # distributions!)
        loss_ll = []  # Loss for each distribution will be appended here
        # Append covariates losses
        for index_type, type in enumerate(self.feat_dists):
            dist, num_params = type
            if dist == 'gaussian':
                mean = inputs[:, index_x]
                std = inputs[:, index_x + 1]
                ll = - torch.log(np.sqrt(2 * np.pi) * std) - 0.5 * ((targets[:, index_type] - mean) / std).pow(2)
            elif dist == 'bernoulli':
                p = inputs[:, index_x]
                ll = targets[:, index_type] * torch.log(p) + (1 - targets[:, index_type]) * torch.log(1 - p)
            elif dist == 'categorical':
                p = inputs[:, index_x: index_x + num_params]
                mask = F.one_hot(targets[:, index_type].long(), num_classes=num_params)  # These are the indexes
                # whose losses we want to compute
                ll = torch.log(p) * mask
            else:
                raise NotImplementedError('Unknown distribution to compute loss')
            check_nan_inf(ll, 'Covariates loss')
            if 0 in imp_mask:
                if dist == 'categorical':
                    ll *= imp_mask[:, index_type].unsqueeze(1)
                else:
                    ll *= imp_mask[:, index_type]  # Add the imputation effect: do NOT train on outputs with mask=0!
            loss_ll.append(-torch.sum(ll) / inputs.shape[0])
            index_x += num_params

        return sum(loss_ll)

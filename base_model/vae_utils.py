# Author: Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 16/03/2023

# Import libraries
import math
import torch

import numpy as np


# -----------------------------------------------------------
#                      TRAINING PROCESS
# -----------------------------------------------------------

def check_nan_inf(values, log):
    if torch.isnan(values).any().detach().cpu().tolist() or torch.isinf(values).any().detach().cpu().tolist():
        raise RuntimeError('NAN DETECTED. ' + str(log))
    return


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.stop = False

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
        if self.counter >= self.patience:
            self.stop = True


def get_dim_from_type(feat_dists):
    return sum(d[1] for d in feat_dists)  # Returns the number of parameters needed
    # to base_model the distributions in feat_dists


def get_activations_from_types(x, feat_dists, min_val=1e-3, max_std=10.0, max_alpha=2, max_k=1000.0):
    # Ancillary function that gives the correct torch activations for each data distribution type
    # Example of type list: [('bernoulli', 1), ('gaussian', 2), ('categorical', 5)]
    # (distribution, number of parameters needed for it)
    index_x = 0
    out = []
    for index_type, type in enumerate(feat_dists):
        dist, num_params = type
        if dist == 'gaussian':
            out.append(x[:, index_x, np.newaxis])  # Mean: from -inf to +inf
            out.append((torch.sigmoid(x[:, index_x + 1, np.newaxis]) * (max_std - min_val) + min_val) / (10 * max_std))
        elif dist == 'bernoulli':
            out.append(torch.sigmoid(x[:, index_x, np.newaxis]) * (1.0 - 2 * min_val) + min_val)
            # p: (min_val, 1-min_val)
        elif dist == 'categorical':  # Softmax activation: NANs appear if values are close to 0,
            # so use min_val to prevent that
            vals = torch.tanh(x[:, index_x: index_x + num_params]) * 10.0  # Limit the max values
            out.append(torch.softmax(vals, dim=1))  # probability of each categorical value
            check_nan_inf(out[-1], 'Categorical distribution')
        else:
            raise NotImplementedError('Distribution ' + dist + ' not implemented')
        index_x += num_params
    return torch.cat(out, dim=1)

# -----------------------------------------------------------
#                      RECONSTRUCTION PROCESS
# -----------------------------------------------------------
def sample_from_dist(params, feat_dists, mode='sample'):  # Get samples from the base_model
    i = 0
    out_vals = []
    for type in feat_dists:
        dist, num_params = type
        if dist == 'gaussian':
            if mode == 'sample':
                x = np.random.normal(loc=params[:, i], scale=params[:, i + 1])
                out_vals.append(x)
            elif mode == 'mode' or mode == 'mean':
                out_vals.append(params[:, 1])  # Mean / mode of the distribution
        elif dist == 'bernoulli':
            if mode == 'sample':
                out_vals.append(np.random.binomial(n=np.ones_like(params[:, i]).astype(int), p=params[:, i]))
            elif mode == 'mode':
                out_vals.append((params[:, 1] > 0.5).astype(int))
            elif mode == 'mean':
                out_vals.append(params[:, 1])
        elif dist == 'categorical':
            if mode == 'sample':
                aux = np.zeros((params.shape[0],))
                for j in range(params.shape[0]):
                    aux[j] = np.random.choice(np.arange(num_params), p=params[j, i: i + num_params])  # Choice
                    # takes p as vector only: we must proceed one by one
                out_vals.append(aux)
            elif mode == 'mode':
                raise NotImplementedError
            elif mode == 'mean':
                raise NotImplementedError
        i += num_params
    return np.array(out_vals).T

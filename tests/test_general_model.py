import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from base_model.vae_model import VariationalAutoencoder
from datasets.external import TableEncoder


def model_params(**options):
    return dict(input_dim=2, hidden_size=8, latent_dim=2,
                feat_distributions=[('gaussian', 2), ('bernoulli', 1)],
                early_stop=False, **options)


def test_preprocessing_uses_training_only_and_masks_unknown_values():
    training = pd.DataFrame({'number': [1.0, 3.0, np.nan], 'category': ['a', 'b', 'a']})
    encoder = TableEncoder().fit(training)
    encoded, mask = encoder.transform(pd.DataFrame({'number': [100.0], 'category': ['unseen']}))
    assert encoder.features[0]['mean'] == 2.0
    assert encoded[0, 0] == 98.0
    assert mask.tolist() == [[1.0, 0.0]]
    assert encoder.features[1]['categories'] == ['a', 'b']
    json.dumps(encoder.features)


def test_stable_profile_validates_once_and_restores_best_weights(monkeypatch):
    model = VariationalAutoencoder(model_params(training_profile='stable'))
    losses = iter([3.0, 1.0, 2.0])

    def epoch(*args, **kwargs):
        loss = next(losses)
        with torch.no_grad():
            model.Encoder.enc1.weight.fill_(loss)
        return dict(loss_tr=loss, loss_va=loss, kl_tr=0, kl_va=0, ll_cov_tr=loss, ll_cov_va=loss)

    monkeypatch.setattr(model, 'fit_epoch', epoch)
    x = np.ones((6, 2), dtype=np.float32)
    model.fit((x, x, x, x), dict(lr=1e-3, n_epochs=3, batch_size=2, device='cpu'))
    assert torch.all(model.Encoder.enc1.weight == 1.0)


@pytest.mark.parametrize('profile,expected_calls', [('paper', 6), ('stable', 4)])
def test_validation_frequency(profile, expected_calls):
    model = VariationalAutoencoder(model_params(training_profile=profile))
    calls = []
    model.register_forward_hook(lambda *args: calls.append(1))
    x = np.ones((6, 2), dtype=np.float32)
    model.fit_epoch((x, x, x, x), torch.optim.Adam(model.parameters()), batch_size=2)
    assert len(calls) == expected_calls


@pytest.mark.parametrize('options', [[], ['--profile', 'stable', '--warmup-epochs', '2',
                                        '--gaussian-std-mode', 'softplus', '--bgm-posterior-samples', '3']])
def test_external_cli_saved_model_reproduces_samples(tmp_path, options):
    rng = np.random.RandomState(0)
    data = pd.DataFrame({'number': rng.normal(size=80), 'group': np.tile(['a', 'b'], 40)})
    data.loc[0, 'number'] = np.nan
    csv = tmp_path / 'data.csv'
    data.to_csv(csv, index=False)
    output = tmp_path / 'model'
    subprocess.run([sys.executable, '-m', 'data_generation.external', 'train', '--input', str(csv),
                    '--output', str(output), '--epochs', '2', '--samples', '20'] + options, check=True)
    reloaded = tmp_path / 'reloaded.csv'
    subprocess.run([sys.executable, '-m', 'data_generation.external', 'sample', '--model', str(output),
                    '--output', str(reloaded), '--samples', '20'], check=True)
    pd.testing.assert_frame_equal(pd.read_csv(output / 'synthetic.csv'), pd.read_csv(reloaded))


def test_mmd_handles_constant_data_and_uses_one_kernel_bandwidth():
    from validation import maximum_mean_discrepancy
    constant = pd.DataFrame(np.zeros((6, 2)))
    assert maximum_mean_discrepancy(constant, constant) == pytest.approx(0.0)
    shifted = pd.DataFrame(np.ones((6, 2)))
    assert maximum_mean_discrepancy(constant, shifted) > 0


def test_latent_bounds_and_old_state_dict_compatibility():
    old = VariationalAutoencoder(model_params())
    bounded = VariationalAutoencoder(model_params(logvar_bounds=[-12.0, 4.0]))
    bounded.load_state_dict(old.state_dict())
    with torch.no_grad():
        bounded.Encoder.enc2.weight.zero_()
        bounded.Encoder.enc2.bias.fill_(100.0)
    logvar = bounded(torch.ones(3, 2))['latent_params'][1]
    assert torch.all(logvar <= 4.0)

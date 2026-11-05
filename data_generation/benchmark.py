"""Paired experiments using the paper's preprocessing, training and evaluation."""
import argparse
import json
import os
import pickle
import platform
import shutil
import time
from importlib.metadata import version
from pathlib import Path
from threadpoolctl import threadpool_limits

import numpy as np
import pandas as pd
import torch

from data_generation.main_generator import train, evaluate
from utils import run_args, preprocess_data


PROFILES = {
    'paper': {},
    'training': {'training_profile': 'stable'},
    'stability': {'training_profile': 'stable', 'grad_clip_norm': 5.0,
                  'logvar_bounds': [-12.0, 4.0]},
    'warmup': {'training_profile': 'stable', 'grad_clip_norm': 5.0,
               'logvar_bounds': [-12.0, 4.0], 'warmup_epochs': 50},
    'decoder': {'training_profile': 'stable', 'grad_clip_norm': 5.0,
                'logvar_bounds': [-12.0, 4.0], 'gaussian_std_mode': 'softplus', 'dropout_p': 0.0},
    'posterior': {'bgm_posterior_samples': 5},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=['adult', 'metabric', 'std'],
                        default=['adult', 'metabric', 'std'])
    parser.add_argument('--profiles', nargs='+', choices=PROFILES, default=['paper'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--output', type=Path, default=Path('experiments/benchmark'))
    parser.add_argument('--summary', action='store_true', help='Summarize completed runs without training')
    parser.add_argument('--reevaluate', action='store_true', help='Repeat evaluation from saved VAE weights')
    options = parser.parse_args()
    if options.epochs < 1 or min(options.seeds) < 0:
        parser.error('epochs must be positive and seeds non-negative')
    options.output.mkdir(parents=True, exist_ok=True)
    if options.summary:
        records = [json.loads(path.read_text()) for path in options.output.glob('*/*/seed_*/metrics.json')]
        if not records:
            parser.error('No completed experiments found')
        frame = pd.DataFrame(records)
        summary = frame.groupby(['dataset', 'profile']).agg(
            seeds=('seed', 'count'), rf_mean=('rf_accuracy', 'mean'), rf_std=('rf_accuracy', 'std'),
            quality_mean=('quality', 'mean'), utility_mean=('utility_synthetic', 'mean'))
        print(summary.to_string())
        summary.to_csv(options.output / 'summary.csv')
        return
    torch.set_num_threads(1)
    args = run_args()
    args.update(n_epochs=options.epochs, gauss=False, mmd=False, train=True, eval=True)
    environment = {'python': platform.python_version(),
                   'packages': {name: version(name) for name in
                                ['numpy', 'pandas', 'torch', 'scikit-learn', 'sdv', 'pycox', 'scipy']}}
    (options.output / 'environment.json').write_text(json.dumps(environment, indent=2))
    for dataset in options.datasets:
        cache = options.output / (dataset + '_data.pkl')
        if cache.exists():
            with cache.open('rb') as handle:
                manager = pickle.load(handle)
        else:
            manager = preprocess_data(dataset, args)
            manager.generate_mask = False
            manager.split_data()
            with cache.open('wb') as handle:
                pickle.dump(manager, handle)
        for profile in options.profiles:
            params = dict(latent_dim=5, hidden_size=50, **PROFILES[profile])
            for seed in options.seeds:
                directory = options.output / dataset / profile / ('seed_' + str(seed))
                directory.mkdir(parents=True, exist_ok=True)
                result_file = directory / 'metrics.json'
                if result_file.exists():
                    previous = json.loads(result_file.read_text())
                    if previous['epochs_limit'] != options.epochs or previous['parameters'] != params:
                        parser.error('Existing experiment uses different settings. Choose another --output directory')
                    if not options.reevaluate:
                        print('Already completed', result_file, flush=True)
                        continue
                args['n_seeds'] = max(options.seeds) + 1
                args['param_comb'] = [params]
                model_dir = directory / 'bgm' / '5_50' / ('seed_' + str(seed))
                model_dir.mkdir(parents=True, exist_ok=True)
                paper_model = (options.output / dataset / 'paper' / ('seed_' + str(seed)) /
                               'bgm' / '5_50' / ('seed_' + str(seed)) / 'model')
                metadata_file = model_dir / 'model.pickle'
                if profile == 'posterior' and not metadata_file.exists() and paper_model.with_suffix('.pickle').exists():
                    shutil.copyfile(paper_model, model_dir / 'model')
                    shutil.copyfile(paper_model.with_suffix('.pickle'), metadata_file)
                args['train_vae'] = not ((model_dir / 'model').exists() and metadata_file.exists())
                if not args['train_vae']:
                    with metadata_file.open('rb') as handle:
                        saved_params = pickle.load(handle)
                    requested = {key: value for key, value in params.items() if not key.startswith('bgm_')}
                    requested['n_epochs'] = options.epochs
                    if any(saved_params.get(key) != value for key, value in requested.items()):
                        parser.error('Saved VAE uses different settings. Choose another --output directory')
                started = time.monotonic()
                result = train(manager, params, seed, str(directory) + os.sep, args)
                manager.set_results_dictionaries([result], args)
                np.random.seed(seed)
                torch.manual_seed(seed)
                scores = evaluate(manager, params, seed, args)
                with (directory / 'evaluation.pkl').open('wb') as handle:
                    pickle.dump(scores, handle)
                utility = scores['task']['cov_comp']
                metrics = {'dataset': dataset, 'profile': profile, 'seed': seed,
                           'epochs_limit': options.epochs, 'parameters': params,
                           'trained_in_run': args['train_vae'],
                           'rf_accuracy': float(scores['disc']['cov_comp']['RF'][1]),
                           'quality': float(scores['stats']['cov_comp'].get_score()),
                           'utility_real': float(utility['real_real'][1]),
                           'utility_synthetic': float(utility['gen_real'][1]),
                           'environment': environment,
                           'seconds': time.monotonic() - started}
                result_file.write_text(json.dumps(metrics, indent=2))
                print(json.dumps(metrics), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()

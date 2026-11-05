"""Train VAE-BGM on a CSV, or sample a saved model."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from data_generation.generator import Generator
from datasets.external import TableEncoder


def generate(model, encoder, samples, seed):
    if samples < 1:
        raise ValueError('samples must be positive')
    np.random.seed(seed)
    torch.manual_seed(seed)
    model.eval()
    model.gen_info = None
    with torch.no_grad():
        model.generate(samples)
    return encoder.inverse_transform(model.gen_info['cov_samples'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    train = commands.add_parser('train', help='Fit preprocessing and model on a training split')
    train.add_argument('--input', required=True, type=Path)
    train.add_argument('--schema', type=Path, help='JSON mapping column names to distributions')
    train.add_argument('--output', type=Path, default=Path('external_output'))
    train.add_argument('--profile', choices=['paper', 'stable'], default='paper')
    train.add_argument('--epochs', type=int, default=1000)
    train.add_argument('--batch-size', type=int, default=500)
    train.add_argument('--latent-dim', type=int, default=5)
    train.add_argument('--hidden-size', type=int, default=50)
    train.add_argument('--learning-rate', type=float, default=1e-3)
    train.add_argument('--beta', type=float, default=1.0, help='KL loss weight')
    train.add_argument('--warmup-epochs', type=int, default=0)
    train.add_argument('--dropout', type=float, default=0.2)
    train.add_argument('--gaussian-std-mode', choices=['paper', 'softplus'], default='paper')
    train.add_argument('--validation-fraction', type=float, default=0.2)
    train.add_argument('--bgm-posterior-samples', type=int, default=0)
    train.add_argument('--samples', type=int, default=1000)
    train.add_argument('--seed', type=int, default=0)
    sample = commands.add_parser('sample', help='Load a trusted model directory and generate CSV')
    sample.add_argument('--model', required=True, type=Path)
    sample.add_argument('--output', required=True, type=Path)
    sample.add_argument('--samples', type=int, default=1000)
    sample.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32:
        parser.error('seed must be an integer in [0, 2**32)')
    torch.set_num_threads(1)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.command == 'sample':
        config = json.loads((args.model / 'config.json').read_text())
        encoder = TableEncoder()
        encoder.features = config['features']
        model = Generator(config['model'])
        model.load(str(args.model / 'weights.pt'))
        # scikit-learn estimators require pickle. Only load your own model artifacts.
        with (args.model / 'bgm.pkl').open('rb') as handle:
            model.bgm = pickle.load(handle)
        generate(model, encoder, args.samples, args.seed).to_csv(args.output, index=False)
        return
    if min(args.epochs, args.batch_size, args.latent_dim, args.hidden_size, args.samples) < 1:
        parser.error('epochs, batch size, dimensions and samples must be positive')
    if not 0 < args.validation_fraction < 1 or args.learning_rate <= 0:
        parser.error('validation fraction must be between 0 and 1 and learning rate positive')
    if not 0 <= args.dropout < 1 or args.beta < 0 or args.warmup_epochs < 0:
        parser.error('dropout must be in [0, 1), beta and warmup epochs must be non-negative')
    if args.warmup_epochs and args.profile == 'paper':
        parser.error('Use --profile stable with KL warmup so validation has a fixed KL weight')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory must be empty to preserve existing experiments')
    frame = pd.read_csv(args.input)
    if len(frame) < 5:
        parser.error('At least five rows are required')
    training, validation = train_test_split(frame, test_size=args.validation_fraction, random_state=args.seed)
    if len(training) < 2:
        parser.error('The training split must contain at least two rows')
    schema = json.loads(args.schema.read_text()) if args.schema else None
    encoder = TableEncoder(schema).fit(training)
    x_train, mask_train = encoder.transform(training)
    x_val, mask_val = encoder.transform(validation)
    params = dict(input_dim=len(frame.columns), latent_dim=args.latent_dim, hidden_size=args.hidden_size,
                  feat_distributions=encoder.distributions, early_stop=True,
                  training_profile='paper' if args.profile == 'paper' else 'stable',
                  beta=args.beta, warmup_epochs=args.warmup_epochs, dropout_p=args.dropout,
                  gaussian_std_mode=args.gaussian_std_mode,
                  bgm_posterior_samples=args.bgm_posterior_samples)
    if args.profile == 'stable':
        params.update(grad_clip_norm=5.0, logvar_bounds=[-12.0, 4.0])
    model = Generator(params)
    history = model.fit((x_train, mask_train, x_val, mask_val),
                        dict(n_epochs=args.epochs, batch_size=args.batch_size,
                             device=torch.device('cpu'), lr=args.learning_rate))
    with torch.no_grad():
        model.train_latent_generator(x_train)
    synthetic = generate(model, encoder, args.samples, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output / 'weights.pt'))
    with (args.output / 'bgm.pkl').open('wb') as handle:
        pickle.dump(model.bgm, handle)
    config = {'model': params, 'features': encoder.features, 'seed': args.seed,
              'training_rows': len(training), 'validation_rows': len(validation),
              'epochs': args.epochs, 'batch_size': args.batch_size, 'learning_rate': args.learning_rate}
    (args.output / 'config.json').write_text(json.dumps(config, indent=2))
    (args.output / 'history.json').write_text(json.dumps(history, indent=2))
    synthetic.to_csv(args.output / 'synthetic.csv', index=False)
    print('Saved model and synthetic.csv in', args.output)


if __name__ == '__main__':
    main()

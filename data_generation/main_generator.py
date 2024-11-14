# Author: Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 13/02/2024


# Packages to import
import os
import sys
import torch

from colorama import Fore, Style
from joblib import Parallel, delayed

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from data_generation.generator import Generator
from utils import run_args, create_output_dir, check_file, save, preprocess_data
from validation import discriminative_validation, ctgan_score_validation, utility_validation


# Function to sort the list of tuples by its third item (mean acc)
def sort_tuple(tup, index=2):
    # getting length of list of tuples
    lst = len(tup)
    for i in range(0, lst):
        for j in range(0, lst - i - 1):
            if (tup[j][index] > tup[j + 1][index]):
                temp = tup[j]
                tup[j] = tup[j + 1]
                tup[j + 1] = temp
    return tup


def get_best_seed_disc_results(results, clas, mode, args, best_seeds=3):
    clas_best_results = {'param_comb': None, 'seeds_acc': [], 'best_seeds_acc': [], 'best_avg_acc': 100,
                         'best_seeds_f1_score': [], 'best_avg_f1_score': 1}
    for params in args['param_comb']:
        model_params = str(params['latent_dim']) + '_' + str(params['hidden_size'])
        seeds_acc = [(seed, seed_results['disc'][mode][clas][0], seed_results['disc'][mode][clas][1],
                      seed_results['disc'][mode][clas][2]) for
                     seed, seed_results in results[model_params].items()]
        best_seeds_acc = sort_tuple(seeds_acc)[:best_seeds]
        avg_acc = [sum(ele) / len(best_seeds_acc) for ele in zip(*best_seeds_acc)][2]
        if avg_acc < clas_best_results['best_avg_acc']:
            clas_best_results['param_comb'] = model_params
            clas_best_results['seeds_acc'] = seeds_acc
            clas_best_results['best_seeds_acc'] = best_seeds_acc
            clas_best_results['best_avg_acc'] = avg_acc
            best_seeds_f1_score = []
            for seed in best_seeds_acc:
                best_seeds_f1_score.append((seed[0], results[model_params][seed[0]]['disc'][mode][clas][5]))
            avg_f1_score = sum(ele[1] for ele in best_seeds_f1_score) / len(best_seeds_f1_score)
            clas_best_results['best_seeds_f1_score'] = best_seeds_f1_score
            clas_best_results['best_avg_f1_score'] = avg_f1_score

    return clas_best_results


def show_utility_results(results, best_rf_results, dataset_name, mode, args):
    utility_best_results = {'param_comb': best_rf_results['param_comb'], 'avg_metric': 1,
                            'best_seeds_metric': []}
    print('Best hyperparameters: ' + str(utility_best_results['param_comb']))
    best_seeds = [seed[0] for seed in best_rf_results['best_seeds_acc']]
    for seed in best_seeds:
        res_seed = results[best_rf_results['param_comb']][seed]['task'][mode]
        utility_best_results['best_seeds_metric'].append((seed, res_seed['real_real'], res_seed['gen_real']))

    sum_metric = 0.0
    for ele in utility_best_results['best_seeds_metric']:
        sum_metric += ele[2][1]
    utility_best_results['avg_metric'] = sum_metric / len(utility_best_results['best_seeds_metric'])
    print('Average metric from best seeds: ' + str(utility_best_results['avg_metric']))
    for seed in utility_best_results['best_seeds_metric']:
        if dataset_name in args['sa_datasets']:
            print('(Real/Gen) Seed ' + str(seed[0]) + ': '
                  + str(format(seed[1][0], '.2f')) + ' - '
                  + str((format(seed[1][1], '.2f'))) + ' - '
                  + str((format(seed[1][2], '.2f'))) + ' / '
                  + str(format(seed[2][0], '.2f')) + ' - '
                  + str((format(seed[2][1], '.2f'))) + ' - '
                  + str((format(seed[2][2], '.2f'))))
        elif dataset_name in args['cl_datasets']:
            print('(Real/Gen) Seed ' + str(seed[0]) + ': '
                  + str(format(seed[1][0], '.2f')) + ' - '
                  + str((format(seed[1][1], '.2f'))) + ' - '
                  + str((format(seed[1][2], '.2f'))) + ' / '
                  + str(format(seed[2][0], '.2f')) + ' - '
                  + str((format(seed[2][1], '.2f'))) + ' - '
                  + str((format(seed[2][2], '.2f'))))
        elif dataset_name in args['reg_datasets']:
            print('(Real-Gen) Seed ' + str(seed[0]) + ': ' + str(
                format(seed[1][1], '.2f')) + ' - ' + str(
                format(seed[2][1], '.2f')))


def show_sdv_best_reports(results, best_rf_results, mode):
    sdv_best_results = {'param_comb': best_rf_results['param_comb'], 'avg_score': 0.0, 'best_seeds_score': []}
    print('Best hyperparameters: ' + str(sdv_best_results['param_comb']))
    best_seeds = [seed[0] for seed in best_rf_results['best_seeds_acc']]
    for seed in best_seeds:
        res_seed = results[best_rf_results['param_comb']][seed]['stats'][mode].get_score()
        sdv_best_results['best_seeds_score'].append((seed, res_seed))

    sum_metric = 0.0
    for ele in sdv_best_results['best_seeds_score']:
        sum_metric += ele[1]
    sdv_best_results['avg_score'] = sum_metric / len(sdv_best_results['best_seeds_score'])
    print('Average score from best seeds: ' + str(sdv_best_results['avg_score']))
    for seed in sdv_best_results['best_seeds_score']:
        print('Seed ' + str(seed[0]) + ': ' + str(format(seed[1], '.2f')))


def train(data_manager, params, seed, output_dir, args):
    # Model parameters
    data = data_manager.model_data
    latent_dim = params['latent_dim']
    hidden_size = params['hidden_size']
    model_params = {'feat_distributions': data_manager.feat_distributions,
                    'latent_dim': latent_dim,
                    'hidden_size': hidden_size,
                    'input_dim': data[0].shape[1],
                    'early_stop': args['early_stop']}
    model_path = str(latent_dim) + '_' + str(hidden_size) + os.sep + 'seed_' + str(seed)
    log_name = output_dir + 'bgm' + os.sep + model_path + os.sep + 'model'
    model = Generator(model_params)

    # Train the base_model
    if args['train_vae']:
        train_params = {'n_epochs': args['n_epochs'],
                        'batch_size': args['batch_size'],
                        'device': torch.device('cpu'),
                        'lr': args['lr'],
                        'path_name': log_name}
        training_results = model.fit(data, train_params)

        # Save base_model information
        model.save(log_name)
        model_params.update(train_params)
        model_params.update(training_results)
        save(model_params, log_name + '.pickle')

    else:
        # Load already trained VAE model
        model.load_state_dict(torch.load(log_name))

    # Obtain and save reconstructed samples using TESTING data
    rec_info = model.predict(data[2])

    # Obtain and save synthetic samples using TESTING data
    n_gen = data[0].shape[0]
    model.train_latent_generator(data[0])
    model.generate(n_gen=n_gen)
    model.validate_samples_range(data_manager, n_gen)

    # Save data generated as csv
    gen_data = data_manager.save_data_to_csv(output_dir, model_path, model.gen_info)

    # Gaussian generation too and compare latent spaces
    if args['gauss']:
        model.generate_gauss(n_gen=n_gen)
        model.validate_samples_range(data_manager, n_gen, gauss=True)

        # Plot latent spaces
        model.compare_latent_spaces(data[0], output_dir + 'gauss' + os.sep + model_path + os.sep)
        gauss_gen_data = data_manager.save_gauss_data_to_csv(output_dir, model_path, model.gauss_gen_info)
        return params, seed, rec_info, model.gen_info, gen_data, model.gauss_gen_info, gauss_gen_data

    return params, seed, rec_info, model.gen_info, gen_data


def evaluate(data_manager, params, seed, args, model='bgm'):
    # Load already preprocessed data
    p_name = str(params['latent_dim']) + '_' + str(params['hidden_size'])
    imp_real_df = data_manager.model_data[0]
    comp_gen_df = data_manager.gen_info[p_name][seed]['cov_samples'] if model == 'bgm' else \
        data_manager.gauss_gen_info[p_name][seed]['cov_samples']

    # Denormalize
    imp_real_df = data_manager.transform_data(imp_real_df, denorm=True)
    comp_gen_df = data_manager.transform_data(comp_gen_df, denorm=True)

    validation = ['cov_comp']

    # Dictionary to save results
    results = {'task': {}, 'disc': {}, 'stats': {}, 'params': params, 'seed': seed}
    classifier_list = args['classifiers_list']

    # Validation process depending on data
    for mode in validation:
        if mode == 'cov_comp':
            results['disc'][mode] = discriminative_validation(imp_real_df, comp_gen_df, data_manager, classifier_list)
            results['task'][mode] = utility_validation(imp_real_df, comp_gen_df, data_manager, args)
            results['stats'][mode] = ctgan_score_validation(imp_real_df, comp_gen_df, data_manager.metadata)
        else:
            raise ValueError('Validation mode not found')

    return results


def eval_model(data_manager, args, output_dir, model='bgm'):
    model_results = Parallel(n_jobs=args['n_threads'], verbose=10)(
        delayed(evaluate)(data_manager, params, seed, args, model) for params in args['param_comb'] for seed in
        range(args['n_seeds']))

    # Create dictionary to save results
    results = {}
    for params in args['param_comb']:
        model_params = str(params['latent_dim']) + '_' + str(params['hidden_size'])
        results[model_params] = {}
        for seed in range(args['n_seeds']):
            results[model_params][seed] = {}

    for res in model_results:
        params = res['params']
        seed = res['seed']
        model_params = str(params['latent_dim']) + '_' + str(params['hidden_size'])
        results[model_params][seed]['disc'] = res['disc']
        results[model_params][seed]['task'] = res['task']
        results[model_params][seed]['stats'] = res['stats']

    # Save results
    save(results, output_dir + model + os.sep + 'results.pkl')


def main():
    print('\n\n-------- SYNTHETIC DATA GENERATION - GENERATOR --------')

    # Environment configuration
    args = run_args()
    task = 'generation'
    create_output_dir(task, args)

    for dataset_name in args['datasets']:
        print('\n\nDataset: ' + Fore.CYAN + dataset_name + Style.RESET_ALL)
        output_dir = args['output_dir'] + dataset_name + os.sep

        if args['train']:
            # Load and prepare data
            data_manager = preprocess_data(dataset_name, args)
            data_manager.save_input_data_to_csv(output_dir)  # Saved for future use
            data_manager.generate_mask = False
            data_manager.split_data()

            # Train
            results = Parallel(n_jobs=args['n_threads'], verbose=10)(
                delayed(train)(data_manager, params, seed, output_dir, args) for params in
                args['param_comb'] for seed in range(args['n_seeds']))

            # Dictionaries are not correctly updated when parallelization. Therefore, update them after training
            data_manager.set_results_dictionaries(results, args)
            if args['gauss']:
                data_manager.set_gauss_results_dictionaries(results, args)
            save(data_manager, output_dir + 'data_manager.pkl')
        else:
            data_manager = check_file(output_dir + 'data_manager.pkl', 'Data manager not found')

        # Evaluate generated data
        if args['eval']:
            eval_model(data_manager, args, output_dir)

            if args['gauss']:
                eval_model(data_manager, args, output_dir, model='gauss')

        # Show results
        model_results = ['bgm', 'gauss'] if args['gauss'] else ['bgm']
        for model in model_results:
            print('\n\n' + Fore.GREEN + model.upper() + ' MODEL' + Style.RESET_ALL)
            results = check_file(output_dir + model + os.sep + 'results.pkl', 'Results file does not exist.')

            # Best results for each mode
            validation = ['cov_comp']

            for mode in validation:
                print('\n\n----' + mode.upper() + '----')

                # Discriminative validation representation
                print('\n----DISCRIMINATIVE VALIDATION----')
                best_rf_results = None
                for clas in args['classifiers_list']:
                    print('[' + clas + ']')
                    best_results = get_best_seed_disc_results(results, clas, mode, args)
                    if clas == 'RF':  # Save best parameters for utility validation
                        best_rf_results = best_results
                    print('Best hyperparameters: ' + str(best_results['param_comb']))
                    print('Average accuracy from best seeds: ' + str(best_results['best_avg_acc']))
                    print('Average F1 score from best seeds: ' + str(best_results['best_avg_f1_score']))
                    for acc, f1_score in zip(best_results['best_seeds_acc'],
                                             best_results['best_seeds_f1_score']):
                        print(
                            'Seed ' + str(acc[0]) + ' accuracy: ' + str(acc[1]) + ' - ' + str(
                                acc[2]) + ' - ' + str(
                                acc[3]) + ' / F1 score: ' + str(f1_score[1]))

                # Utility validation representation just for best seeds for RF classifier
                if 'cov' in mode and best_rf_results is not None:
                    print('\n----UTILITY VALIDATION----')
                    show_utility_results(results, best_rf_results, dataset_name, mode, args)

                # CTGAN/TVAE score representation
                print('\n----CTGAN/TVAE SCORE----')
                show_sdv_best_reports(results, best_rf_results, mode)





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

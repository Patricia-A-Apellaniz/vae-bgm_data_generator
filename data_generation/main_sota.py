# Author: Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 20/12/2023


# Packages to import
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from colorama import Fore, Style
from sdv.single_table import CTGANSynthesizer
from utils import run_args, create_output_dir, check_file, save
from validation import discriminative_validation, utility_validation, ctgan_score_validation


# https://github.com/sdv-dev/CTGAN
# Modeling Tabular data using Conditional GAN (https://arxiv.org/abs/1907.00503)
def train(data_manager, output_dir, model_name):
    # Create a synthesizer
    if model_name == 'ctgan':
        synthesizer = CTGANSynthesizer(data_manager.metadata, epochs=1000, cuda=False)
    else:
        raise RuntimeError('State-of-the-art model not recognized')

    # Train synthesizer
    synthesizer.fit(data_manager.norm_df)
    # Save synthesizer
    synthesizer.save(output_dir + model_name + '_synthesizer.pkl')
    # Generate samples (generate same amount of data as generator)
    synthetic_df = synthesizer.sample(num_rows=data_manager.model_data[0].shape[0])
    # Save dataframe
    synthetic_df.to_csv(output_dir + 'gen_data.csv', index=False)


def eval(data_manager, synthetic_df, output_dir, args):
    real_df = data_manager.model_data[0]  # Use the same as the generator, even though sota models use the complete
    results = {'disc': discriminative_validation(real_df, synthetic_df, data_manager, args['classifiers_list']),
               'task': utility_validation(real_df, synthetic_df, data_manager, args),
               'stats': ctgan_score_validation(real_df, synthetic_df, data_manager.metadata)}

    # Save results
    save(results, output_dir + 'results.pkl')
    return results


def main():
    print('\n\n-------- SYNTHETIC DATA GENERATION - STATE OF THE ART MODELS --------')

    # Environment configuration
    args = run_args()
    task = 'sota_generation'
    create_output_dir(task, args)

    for dataset_name in args['datasets']:
        print('\n\nDataset: ' + Fore.CYAN + dataset_name + Style.RESET_ALL)
        for model in args['models']:
            print('\nModel: ' + Fore.RED + model + Style.RESET_ALL)
            output_dir = args['sota_output_dir'] + dataset_name + os.sep + model + os.sep

            # Load data_manager object
            data_manager = check_file(args['output_dir'] + dataset_name + os.sep + 'data_manager.pkl',
                                      'Data manager not found. Run data_generation.py first.')

            # Train model
            if args['train']:
                train(data_manager, output_dir, model)

            # Evaluate real vs synthetic data
            if args['eval']:
                # Check if synthetic dataframe exists
                syn_df = check_file(output_dir + 'gen_data.csv', 'Synthetic data not generated.', csv=True)

                # Evaluation
                eval(data_manager, syn_df, output_dir, args)

            # Show results
            results = check_file(output_dir + 'results.pkl', 'Results file does not exist.')

            # Discriminative validation representation
            print('\n----DISCRIMINATIVE VALIDATION----')
            for clas in args['classifiers_list']:
                print('[' + clas + ']')
                print('Accuracy:' + str(results['disc'][clas][0]) + ' - ' + str(
                    results['disc'][clas][1]) + ' - ' + str(
                    results['disc'][clas][2]))
                print('Precision:' + str(results['disc'][clas][3]))
                print('Recall:' + str(results['disc'][clas][4]))
                print('F1 score:' + str(results['disc'][clas][5]))

            # Utility validation representation
            print('\n----UTILITY VALIDATION----')
            if dataset_name in args['cl_datasets']:
                print('Accuracy (real_real/gen_real): ' + str(results['task']['real_real'][0]) + ' - ' + str(
                    results['task']['real_real'][1]) + ' - ' + str(
                    results['task']['real_real'][2]) + ' / ' + str(results['task']['gen_real'][0]) + ' - ' + str(
                    results['task']['gen_real'][1]) + ' - ' + str(
                    results['task']['gen_real'][2]))
                print('Precision (real_real/gen_real): ' + str(results['task']['real_real'][3]) + ' / ' + str(
                    results['task']['gen_real'][3]))
                print('Recall (real_real/gen_real): ' + str(results['task']['real_real'][4]) + ' / ' + str(
                    results['task']['gen_real'][4]))
                print('F1 score (real_real/gen_real): ' + str(results['task']['real_real'][5]) + ' / ' + str(
                    results['task']['gen_real'][5]))
            elif dataset_name in args['sa_datasets']:
                print('[COXPH] C-index (real_real/gen_real): ' + str(results['task']['real_real'][0]) + ' - ' + str(
                    results['task']['real_real'][1]) + ' - ' + str(
                    results['task']['real_real'][2]) + ' / ' + str(results['task']['gen_real'][0]) + ' - ' + str(
                    results['task']['gen_real'][1]) + ' - ' + str(
                    results['task']['gen_real'][2]))

            # CTGAN/TVAE score representation
            print('\n----CTGAN/TVAE SCORE----')
            print('CTGAN score: ' + str(results['stats'].get_score()))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

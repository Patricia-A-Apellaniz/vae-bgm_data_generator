# Author: Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 10/01/2024


# Packages to import
import torch

import numpy as np
import pandas as pd
import torchtuples as tt

from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
from sklearn.model_selection import train_test_split
from sdmetrics.reports.single_table import QualityReport
from statsmodels.stats.proportion import proportion_confint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    ConfusionMatrixDisplay

# This warning type is removed due to pandas future warnings
# https://github.com/havakv/pycox/issues/162. Incompatibility between pycox and pandas' new version
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


# ------------------------------------------------------------------------------------------------------
#                                     UTILS FUNCTIONS
# ------------------------------------------------------------------------------------------------------

def fix_imbalance(real_df, synthetic_df):
    n_syn, n_real = synthetic_df.shape[0], real_df.shape[0]
    n = min((n_syn, n_real))
    real_df = real_df.sample(frac=1, random_state=0).reset_index(drop=True)
    synthetic_df = synthetic_df.sample(frac=1, random_state=0).reset_index(drop=True)
    real_df = real_df[0: n]
    synthetic_df = synthetic_df[0: n]
    return real_df, synthetic_df


def bern_conf_interval(n, mean, acc=True):
    # Confidence interval
    ci_bot, ci_top = proportion_confint(count=mean * n, nobs=n, alpha=0.1, method='beta')
    if mean < 0.5 and acc:
        ci_bot_2 = 1 - ci_top
        ci_top = 1 - ci_bot
        ci_bot = ci_bot_2
        mean = 1 - mean

    return np.round(ci_bot, 4), mean, np.round(ci_top, 4)


# ------------------------------------------------------------------------------------------------------
#                                   CTGAN/TVAE VALIDATION
# ------------------------------------------------------------------------------------------------------
def ctgan_score_validation(real_df, synthetic_df, metadata):
    report = QualityReport()
    report.generate(real_df, synthetic_df, metadata.to_dict(), verbose=False)
    return report


# ------------------------------------------------------------------------------------------------------
#                                   DISCRIMINATIVE VALIDATION
# ------------------------------------------------------------------------------------------------------
def define_classifiers(disc_list):
    classifiers = {}
    for clas in disc_list:
        if clas == 'RF':
            classifiers[clas] = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=3)
        else:
            raise NotImplementedError('Classifier not recognized')
    return classifiers


def discriminative_validation(real_df, synthetic_df, data_manager, classifier_list, split=0.8):
    # Imbalance correction. We need to compare same size datasets
    real_df, synthetic_df = fix_imbalance(real_df, synthetic_df)

    # Add label to identify each dataset
    real_df['target'] = np.zeros(real_df.shape[0]).astype(int)
    synthetic_df['target'] = np.ones(synthetic_df.shape[0]).astype(int)
    mixed_data = [real_df, synthetic_df]
    mixed_data = pd.concat(mixed_data, axis=0).sample(frac=1, random_state=0).reset_index(drop=True)

    # Preprocess data
    x_train, y_train, x_test, y_test = preprocess_ml_data(mixed_data, data_manager, split)

    # Define the classifiers
    classifiers = define_classifiers(classifier_list)
    classifiers_names = [key for key in classifiers.keys()]

    # Train and test the classifiers
    results = {}
    for clas in classifiers_names:
        results[clas] = classification_task(classifiers[clas], x_train, y_train, x_test, y_test)

    return results


# ------------------------------------------------------------------------------------------------------
#                                   SURVIVAL ANALYSIS VALIDATION
# ------------------------------------------------------------------------------------------------------
def obtain_c_index(surv_f, time, censor):
    # Evaluate using PyCox c-index
    ev = EvalSurv(surv_f, time.flatten(), censor.flatten(), censor_surv='km')
    ci = ev.concordance_td()

    # Obtain also ibs
    time_grid = np.linspace(time.min(), time.max(), 100)
    ibs = ev.integrated_brier_score(time_grid)
    return ci, ibs


def preprocess_sa_data(concat_df, real_n_samples, data_manager):
    # First, data imputation
    x = data_manager.impute_data(concat_df, norm=False)

    # Separate real and synthetic data
    real_df = x[:real_n_samples]
    syn_df = x[real_n_samples:]
    real_y = real_df[['time', 'event']]
    real_x = real_df.drop(columns=['time', 'event'])
    syn_y = syn_df[['time', 'event']]
    syn_x = syn_df.drop(columns=['time', 'event'])

    return real_x, real_y, syn_x, syn_y


def survival_analysis(real_df, synthetic_df, data_manager):
    modes = ['real_real', 'gen_real']
    results = {modes[0]: [], modes[1]: []}

    # We should process them together (imputation, normalization if necessary...)
    concat_df = pd.concat([real_df, synthetic_df], axis=0)  # No shuffle to keep the order
    real_x, real_y, syn_x, syn_y = preprocess_sa_data(concat_df, real_df.shape[0], data_manager)

    # Split data
    tr_real_x, te_real_x, tr_real_y, te_real_y = train_test_split(real_x, real_y, test_size=0.2, random_state=0)
    tr_real_x, va_real_x, tr_real_y, va_real_y = train_test_split(tr_real_x, tr_real_y, test_size=0.2, random_state=0)
    tr_syn_x, te_syn_x, tr_syn_y, te_syn_y = train_test_split(syn_x, syn_y, test_size=0.2, random_state=0)
    tr_syn_x, va_syn_x, tr_syn_y, va_syn_y = train_test_split(tr_syn_x, tr_syn_y, test_size=0.2, random_state=0)

    # Train the SA model
    for mode in modes:
        if mode == 'real_real':
            # 1. Train with real data and test with real data
            x_train = tr_real_x.to_numpy().astype('float32')
            y_train = (tr_real_y.iloc[:, -2].values.astype('float32'), tr_real_y.iloc[:, -1].values.astype('float32'))
            x_val = va_real_x.to_numpy().astype('float32')
            y_val = (va_real_y.iloc[:, -2].values.astype('float32'), va_real_y.iloc[:, -1].values.astype('float32'))
            val = (x_val, y_val)
            x_test = te_real_x.to_numpy().astype('float32')
            y_test = (te_real_y.iloc[:, -2].values.astype('float32'), te_real_y.iloc[:, -1].values.astype('float32'))
        elif mode == 'gen_real':
            # 2. Train with syn data and test with real data
            x_train = tr_syn_x.to_numpy().astype('float32')
            y_train = (tr_syn_y.iloc[:, -2].values.astype('float32'), tr_syn_y.iloc[:, -1].values.astype('float32'))
            x_val = va_syn_x.to_numpy().astype('float32')
            y_val = (va_syn_y.iloc[:, -2].values.astype('float32'), va_syn_y.iloc[:, -1].values.astype('float32'))
            val = (x_val, y_val)
            x_test = te_real_x.to_numpy().astype('float32')
            y_test = (te_real_y.iloc[:, -2].values.astype('float32'), te_real_y.iloc[:, -1].values.astype('float32'))
        else:
            raise RuntimeError('[ERROR] Classifying mode not recognized.')

        # Linear model for classical Cox regression
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        in_features = x_train.shape[1]
        batch_size = 256
        net = torch.nn.Linear(in_features, 1)
        optimizer = tt.optim.AdamWR(decoupled_weight_decay=0.01, cycle_eta_multiplier=0.8, cycle_multiplier=2)
        model = CoxPH(net, optimizer, device=torch.device('cpu'))
        model.optimizer.set_lr(0.06)

        # Training
        epochs = 100
        callbacks = [tt.callbacks.EarlyStopping()]
        verbose = False
        _ = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose, val_data=val,
                      val_batch_size=batch_size)

        # Prediction with real data
        _ = model.compute_baseline_hazards(x_train, y_train)
        survival_f = model.predict_surv_df(x_test)

        # Concordance index computation with PyCox
        ci = obtain_c_index(survival_f, y_test[0], y_test[1])[0]
        ci_results = bern_conf_interval(len(y_test[0]), ci)
        results[mode] = ci_results

    return results


# ------------------------------------------------------------------------------------------------------
#                                   ML TASK VALIDATION
# ------------------------------------------------------------------------------------------------------

def preprocess_ml_data(data, data_manager, split, target='target'):
    # Data imputation
    data = data_manager.impute_data(data, norm=False)

    # Transform data to float
    data = data.astype('float64')

    # If target feature is string, convert to categorical
    if data[target].dtype in [object, str]:
        data[target] = pd.factorize(data[target])[0]

    # Data splitting
    train_split = int(len(data) * split)
    train_data = data[0: train_split]
    test_data = data[train_split:].reset_index(drop=True)

    # Target feature
    y_train = np.array(train_data[target].values)
    x_train = np.array(train_data.drop(columns=[target]))
    y_test = np.array(test_data[target].values)
    x_test = np.array(test_data.drop(columns=[target]))

    return x_train, y_train, x_test, y_test


def classification_task(classifier, x_train, y_train, x_test, y_test):
    # Train classifier and predict
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)

    # Metrics
    acc = np.round(accuracy_score(y_test, predictions), 4)
    acc_results = bern_conf_interval(len(predictions), acc)
    acc_bot, acc_mean, acc_top = acc_results
    prec = np.round(precision_score(y_test, predictions, zero_division=0), 4) if np.unique(y_train).shape[
                                                                                     0] <= 2 else np.round(
        precision_score(y_test, predictions, zero_division=0, average='micro'), 4)
    rec = np.round(recall_score(y_test, predictions), 4) if np.unique(y_train).shape[0] <= 2 else np.round(
        recall_score(y_test, predictions, zero_division=0, average='micro'), 4)
    f1 = np.round(f1_score(y_test, predictions), 4) if np.unique(y_train).shape[0] <= 2 else np.round(
        f1_score(y_test, predictions, zero_division=0, average='micro'), 4)
    conf_m = confusion_matrix(y_test, predictions)
    results = [acc_bot, acc_mean, acc_top,
               prec,
               rec,
               f1,
               ConfusionMatrixDisplay(confusion_matrix=conf_m, display_labels=range(0, 2))]
    return results


def ml_analysis(real_df, synthetic_df, data_manager, args, task=None, split=0.8):
    modes = ['real_real', 'gen_real']

    # Preprocess data together
    real_df = real_df.rename(columns={'label': 'target'})
    synthetic_df = synthetic_df.rename(columns={'label': 'target'})
    concat_df = pd.concat([real_df, synthetic_df], axis=0)
    if concat_df.isna().sum().sum() > 0:
        concat_df = data_manager.impute_data(concat_df, norm=False)
    real_df = concat_df[:real_df.shape[0]]
    synthetic_df = concat_df[real_df.shape[0]:]
    tr_real, te_real, tr_syn, te_syn = train_test_split(real_df, synthetic_df, test_size=0.2, random_state=0)
    tr_real_x = tr_real.drop(columns=['target'])
    tr_real_y = tr_real['target']
    te_real_x = te_real.drop(columns=['target'])
    te_real_y = te_real['target']
    tr_syn_x = tr_syn.drop(columns=['target'])
    tr_syn_y = tr_syn['target']

    # Train the classifier
    results = {}
    for i, mode in enumerate(modes):
        results[mode] = []
        if mode == 'real_real':
            # 1. Train with real data and test with real data
            x_train = tr_real_x
            y_train = tr_real_y
            x_test = te_real_x
            y_test = te_real_y
        elif mode == 'gen_real':
            # 2. Train with syn data and test with real data
            x_train = tr_syn_x
            y_train = tr_syn_y
            x_test = te_real_x
            y_test = te_real_y
        else:
            raise RuntimeError('Classifying mode not recognized.')

        if data_manager.dataset_name in args['cl_datasets'] or task == 'cl':
            classifier = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=3, max_depth=3)
            results[mode] = classification_task(classifier, x_train, y_train, x_test, y_test)
    return results


def utility_validation(real_df, synthetic_df, data_manager, args):
    # Imbalance correction. We need to compare same size datasets
    real_df, synthetic_df = fix_imbalance(real_df, synthetic_df)

    dataset_name = data_manager.dataset_name
    if dataset_name in args['sa_datasets']:
        results = survival_analysis(real_df, synthetic_df, data_manager)
    elif dataset_name in args['cl_datasets']:
        results = ml_analysis(real_df, synthetic_df, data_manager, args)
    else:
        raise RuntimeError('Dataset task has not been specified.')

    return results

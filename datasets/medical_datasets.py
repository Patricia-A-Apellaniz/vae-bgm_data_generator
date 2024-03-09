# Author: Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 05/03/2024


# Packages to import
import numpy as np
import pandas as pd

from pycox import datasets
from .data_manager import DataManager
from sdv.metadata import SingleTableMetadata


def preprocess_metabric(dataset_name):
    # Load data
    raw_df = datasets.metabric.read_df()

    # Transform covariates and create df
    label = raw_df[['event']]
    time = raw_df[['duration']]
    raw_df = raw_df.drop(labels=['event', 'duration'], axis=1)
    survival_df = time.join(label)
    raw_df = raw_df.join(survival_df)
    raw_df = raw_df.rename(columns={raw_df.columns[-1]: 'event', raw_df.columns[-2]: 'time'})
    raw_metadata = SingleTableMetadata()
    raw_metadata.detect_from_dataframe(raw_df)

    # Transform covariates and create df
    df = raw_df.copy()

    # Create data manager object
    data_manager = DataManager(dataset_name, raw_df, df, raw_metadata=raw_metadata)

    # Obtain feature distributions
    feat_distributions = []
    for i in range(df.shape[1]):
        values = df.iloc[:, i].unique()
        no_nan_values = values[~pd.isnull(values)]
        if no_nan_values.size <= 2 and np.all(np.sort(no_nan_values).astype(int) ==
                                              np.array(range(no_nan_values.min().astype(int),
                                                             no_nan_values.min().astype(int) + len(no_nan_values)))):
            feat_distributions.append(('bernoulli', 1))
        elif np.amin(np.equal(np.mod(no_nan_values, 1), 0)):
            # Check if values are floats but don't have decimals and transform to int. They are floats because of NaNs
            if no_nan_values.dtype == 'float64':
                no_nan_values = no_nan_values.astype(int)
            if np.unique(no_nan_values).size < 50 and np.amin(no_nan_values) == 0:
                feat_distributions.append(('categorical', (np.max(no_nan_values) + 1).astype(int)))
            else:
                feat_distributions.append(('gaussian', 2))
        else:
            feat_distributions.append(('gaussian', 2))
    data_manager.set_feat_distributions(feat_distributions)

    # Normalize, impute data
    # Necessary to impute before normalization because of the categorical variables treated as gaussian.
    data_manager.norm_df = data_manager.transform_data(df)
    data_manager.imp_norm_df = data_manager.impute_data(data_manager.norm_df)

    # Create metadata for ctgan and tvae
    data_manager.get_metadata()

    return data_manager


def preprocess_std(dataset_name, args):
    # Load data
    data_filename = args['input_dir'] + 'std/std.csv'
    raw_df = pd.read_csv(data_filename, sep=',', index_col=0)

    # Transform covariates and create df
    label = raw_df[['rinfct']]
    time = raw_df[['time']]
    raw_df = raw_df.drop(labels=['obs', 'rinfct', 'time'], axis=1)
    survival_df = time.join(label)
    raw_df = raw_df.join(survival_df)
    raw_df = raw_df.rename(columns={raw_df.columns[-1]: 'event', raw_df.columns[-2]: 'time'})
    raw_metadata = SingleTableMetadata()
    raw_metadata.detect_from_dataframe(raw_df)

    # Transform covariates and create df
    df = raw_df.copy()
    mapping_info = {}
    df['race'], classes = df['race'].factorize()
    mapping_info['race'] = np.array(classes.values)
    df['race'] = df['race'].replace(-1, np.nan)
    df['marital'], classes = df['marital'].factorize()
    mapping_info['marital'] = np.array(classes.values)
    df['marital'] = df['marital'].replace(-1, np.nan)
    df['iinfct'], classes = df['iinfct'].factorize()
    mapping_info['iinfct'] = np.array(classes.values)
    df['iinfct'] = df['iinfct'].replace(-1, np.nan)
    df['condom'], classes = df['condom'].factorize()
    mapping_info['condom'] = np.array(classes.values)
    df['condom'] = df['condom'].replace(-1, np.nan)

    # Create data manager object
    data_manager = DataManager(dataset_name, raw_df, df, mapping_info=mapping_info, raw_metadata=raw_metadata)

    # Obtain feature distributions
    feat_distributions = []
    for i in range(df.shape[1]):
        values = df.iloc[:, i].unique()
        no_nan_values = values[~pd.isnull(values)]
        if no_nan_values.size <= 2 and np.all(np.sort(no_nan_values).astype(int) ==
                                              np.array(range(no_nan_values.min().astype(int),
                                                             no_nan_values.min().astype(int) + len(no_nan_values)))):
            feat_distributions.append(('bernoulli', 1))
        elif np.amin(np.equal(np.mod(no_nan_values, 1), 0)):
            # Check if values are floats but don't have decimals and transform to int. They are floats because of NaNs
            if no_nan_values.dtype == 'float64':
                no_nan_values = no_nan_values.astype(int)
            if np.unique(no_nan_values).size < 50 and np.amin(no_nan_values) == 0:
                feat_distributions.append(('categorical', (np.max(no_nan_values) + 1).astype(int)))
            else:
                feat_distributions.append(('gaussian', 2))
        else:
            feat_distributions.append(('gaussian', 2))
    data_manager.set_feat_distributions(feat_distributions)

    # Normalize, impute data
    # Necessary to impute before normalization because of the categorical variables treated as gaussian.
    data_manager.norm_df = data_manager.transform_data(df)
    data_manager.imp_norm_df = data_manager.impute_data(data_manager.norm_df)

    # Create metadata for ctgan and tvae
    data_manager.get_metadata()

    return data_manager
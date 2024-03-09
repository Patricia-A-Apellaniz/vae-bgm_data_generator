# Author: Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 24/01/2024


# Packages to import
import numpy as np
import pandas as pd

from .data_manager import DataManager
from sdv.datasets.demo import download_demo


# Convert continuous variables to categorical, zero is a class and the rest of the elements are divided into 3
# classes
def cont2cat(df, cols, n=4):
    # Compute the bins for each column
    for col in cols:
        # Compute quartiles
        bins_aux = [df[col].quantile(i / n) for i in range(n + 1)]
        if len(bins_aux) == len(set(bins_aux)):
            df[col], bins = pd.qcut(df[col], q=n, retbins=True, duplicates='drop', labels=False)
        else:
            for i in range(1, len(bins_aux)):
                if bins_aux[i] < bins_aux[i - 1]:
                    bins_aux[i] = bins_aux[i - 1] + 0.0001
                elif bins_aux[i] == bins_aux[i - 1]:
                    bins_aux[i] = bins_aux[i] + 0.0001

            # Create interval index
            df[col] = pd.cut(df[col], bins_aux, labels=False, include_lowest=True, right=True)
            df[col] = df[col].astype('Int64')
            # column should have consecutive integers as categories
            # unique values
            df[col] = df[col].astype('category')
            # label encoding
            df[col] = df[col].astype('category').cat.codes
    return df


def preprocess_adult(dataset_name):
    # Load data
    raw_df, metadata = download_demo(modality='single_table', dataset_name='adult')

    # Transform '?' values to nan values
    raw_df = raw_df.replace('?', np.nan)

    # Transform continuous variables to categorical
    raw_df = cont2cat(raw_df, ['hours-per-week', 'capital-gain', 'capital-loss'])

    # Take just 10.000 samples
    raw_df = raw_df.sample(n=10000, random_state=0).reset_index(drop=True)

    # Drop irrelevant columns
    raw_df = raw_df.drop(labels=['education-num'], axis=1)

    # Remove columns also in metadata
    metadata_cols = metadata.columns
    del metadata_cols['education-num']
    metadata.columns = metadata_cols.copy()

    # Transform covariates and create df
    df = raw_df.copy()
    mapping_info = {}
    df['workclass'], classes = df['workclass'].factorize()
    mapping_info['workclass'] = np.array(classes.values)
    df['workclass'] = df['workclass'].replace(-1, np.nan)
    df['education'], classes = df['education'].factorize()
    mapping_info['education'] = np.array(classes.values)
    df['marital-status'], classes = df['marital-status'].factorize()
    mapping_info['marital-status'] = np.array(classes.values)
    df['occupation'], classes = df['occupation'].factorize()
    mapping_info['occupation'] = np.array(classes.values)
    df['occupation'] = df['occupation'].replace(-1, np.nan)
    df['relationship'], classes = df['relationship'].factorize()
    mapping_info['relationship'] = np.array(classes.values)
    df['race'], classes = df['race'].factorize()
    mapping_info['race'] = np.array(classes.values)
    df['sex'] = df['sex'].apply(lambda x: 0 if x == 'Male' else 1)
    mapping_info['sex'] = np.array(['Male', 'Female'])
    df['native-country'], classes = df['native-country'].factorize()
    mapping_info['native-country'] = np.array(classes.values)
    df['native-country'] = df['native-country'].replace(-1, np.nan)
    df['label'] = df['label'].apply(lambda x: 0 if x == '<=50K' else 1)
    mapping_info['label'] = np.array(['<=50K', '>50K'])

    # Create data manager object
    data_manager = DataManager(dataset_name, raw_df, df, mapping_info, raw_metadata=metadata)

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

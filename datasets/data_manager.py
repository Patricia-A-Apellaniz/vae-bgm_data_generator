# Author: Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 16/01/2024

# Packages to import
import os

import numpy as np
import pandas as pd

from scipy import stats
from sklearn.svm import SVR
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sdv.metadata import SingleTableMetadata
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split


class DataManager:
    def __init__(self, dataset_name, raw_df, processed_df, mapping_info=None, raw_metadata=None):
        self.dataset_name = dataset_name
        self.raw_df = raw_df
        self.processed_df = processed_df
        self.columns = self.processed_df.columns
        self.mapping_info = mapping_info
        self.raw_metadata = raw_metadata
        self.imp_df = None
        self.norm_df = None
        self.imp_norm_df = None
        self.feat_distributions = None
        self.positive_gaussian_cols = None
        self.model_data = None
        self.rec_info = {}
        self.gen_info = {}
        self.generate_mask = False
        self.gen_raw_data = {}
        self.metadata = None
        self.gauss_gen_info = {}
        self.gauss_gen_raw_data = {}

        # Mask to be used during training
        if self.processed_df.isna().any().any():
            nans = self.processed_df.isna()
            self.raw_mask = nans.replace([True, False], [0, 1])

            # Check if generate_mask is True and nan values exist
            self.generate_mask = True
            self.gen_mask = {}
            self.gen_nan_raw_data = {}
            self.gauss_gen_mask = {}
            self.gauss_gen_nan_raw_data = {}
        else:
            mask = np.ones((self.processed_df.shape[0], self.processed_df.shape[1]))
            self.raw_mask = pd.DataFrame(mask, columns=self.columns)

        self.mask = self.raw_mask.copy()

    def set_feat_distributions(self, feat_distributions):
        self.feat_distributions = feat_distributions
        # Get positive gaussian columns for postprocessing purposes
        positive_columns = []
        for idx, dist in enumerate(self.feat_distributions):
            if dist[0] == 'gaussian':
                values = self.processed_df.iloc[:, idx].values.astype(float)
                non_missing_values = values[~np.isnan(values)]
                if not (non_missing_values < 0).any():
                    positive_columns.append(idx)
        self.positive_gaussian_cols = positive_columns

    # Necessary for CTGAN
    def get_metadata(self, metadata=None):
        if metadata is None:
            self.metadata = SingleTableMetadata()
            self.metadata.detect_from_dataframe(self.norm_df)
        else:
            self.metadata = metadata
        return self.metadata

    # Save processed_df as .csv file. This file has the input data for data generation models and validation
    # processes
    def save_input_data_to_csv(self, path):
        self.processed_df.to_csv(path + 'preprocessed_data.csv', index=False)

    # Concatenate mask to data to generate synthetic missing positions too
    def concat_mask(self):
        mask_names = ['imp_mask_' + col for col in self.columns]
        mask_copy = self.raw_mask.copy()
        mask_copy.columns = mask_names
        self.imp_norm_df = pd.concat([self.imp_norm_df, mask_copy], axis=1)
        mask_extension_df = self.raw_mask.copy()
        mask_extension_df.columns = ['mask_ext_' + col for col in self.columns]
        # It is necessary to concatenate ones mask to imputation mask for training purposes
        self.mask = pd.concat([mask_copy, mask_extension_df.replace(0, 1)], axis=1)
        # Get new data distributions. Mask features should be bernoulli!
        self.feat_distributions.extend(
            [('bernoulli', 1) for _ in range(self.mask.shape[1] - len(self.feat_distributions))])

    def split_data(self, split=0.2):
        # Split data
        train_data, test_data, train_mask, test_mask = train_test_split(self.imp_norm_df, self.mask, test_size=split,
                                                                        random_state=0)
        self.model_data = (train_data.reset_index(drop=True),
                           train_mask.reset_index(drop=True),
                           test_data.reset_index(drop=True),
                           test_mask.reset_index(drop=True))

    def zero_imputation(self, data):
        imp_data = data.copy()
        imp_data = imp_data.fillna(0)
        return imp_data

    def mice_imputation(self, data, model='bayesian'):
        imp_data = data.copy()
        if model == 'bayesian':
            clf = BayesianRidge()
        elif model == 'svr':
            clf = SVR()
        else:
            raise RuntimeError('MICE imputation base_model not recognized')
        imp = IterativeImputer(estimator=clf, verbose=2, max_iter=30, tol=1e-10, imputation_order='roman')
        imp_data.iloc[:, :] = imp.fit_transform(imp_data)
        return imp_data

    def statistics_imputation(self, data, norm):
        imp_data = data.copy()
        # If data comes from classification task, columns size doesn't match data's columns size
        n_columns = data.columns.size if data.columns.size < self.columns.size else self.columns.size
        for i in range(n_columns):
            values = data.iloc[:, i].values
            raw_values = self.processed_df.iloc[:, i].values
            if any(pd.isnull(values)):
                no_nan_values = values[~pd.isnull(values)]
                no_nan_raw_values = raw_values[~pd.isnull(raw_values)]
                if values.dtype in [object, str] or no_nan_values.size <= 2 or np.amin(
                        np.equal(np.mod(no_nan_values, 1), 0)):
                    stats_value = stats.mode(no_nan_values, keepdims=True)[0][0]
                # If raw data has int values take mode normalized
                elif norm and np.amin(np.equal(np.mod(no_nan_raw_values, 1), 0)):
                    stats_value = stats.mode(no_nan_raw_values, keepdims=True)[0][0]
                    # Find index of stats_value in self.raw_df.iloc[:, i].values
                    idx = np.where(self.processed_df.iloc[:, i].values == stats_value)[0][0]
                    # Find which value is in idx of data.iloc[:, i].values and set this value to stats_value
                    stats_value = values[np.where(values == data.iloc[:, i].values[idx])[0][0]]
                else:
                    stats_value = no_nan_values.mean()
                imp_data.iloc[:, i] = [stats_value if pd.isnull(x) else x for x in imp_data.iloc[:, i]]

        return imp_data

    # Transform data according to raw_df
    def transform_data(self, df, denorm=False):
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        transformed_df = df.copy()
        for i in range(self.processed_df.shape[1]):
            dist = self.feat_distributions[i][0]
            values = self.processed_df.iloc[:, i]
            no_nan_values = values[~pd.isnull(values)].values
            if dist == 'gaussian':
                loc = np.mean(no_nan_values)
                scale = np.std(no_nan_values)
            elif dist == 'bernoulli':
                loc = np.amin(no_nan_values)
                scale = np.amax(no_nan_values) - np.amin(no_nan_values)
            elif dist == 'categorical':
                loc = np.amin(no_nan_values)
                scale = 1  # Do not scale
            else:
                raise NotImplementedError('Distribution ', dist, ' not normalized!')

            if denorm:  # Denormalize
                transformed_df.iloc[:, i] = (df.iloc[:, i] * scale + loc if scale != 0
                                             else df.iloc[:, i] + loc).astype(self.processed_df.iloc[:, i].dtype)
            else:  # Normalize
                transformed_df.iloc[:, i] = (df.iloc[:, i] - loc) / scale if scale != 0 else df.iloc[:, i] - loc

        return transformed_df

    def impute_data(self, df, mode='stats', norm=True):
        # If missing data exists, impute it
        if df.isna().any().any():
            # Data imputation
            if mode == 'zero':
                imp_df = self.zero_imputation(df)
            elif mode == 'stats':
                imp_df = self.statistics_imputation(df, norm)
            else:
                imp_df = self.mice_imputation(df)
        else:
            imp_df = df.copy()

        return imp_df

    # Since several parameters and seeds are used, we need to save the results in a dictionary
    def set_results_dictionaries(self, results, args):
        # Create results dictionaries
        for params in args['param_comb']:
            p_name = str(params['latent_dim']) + '_' + str(params['hidden_size'])
            self.rec_info[p_name] = {}
            self.gen_info[p_name] = {}
            self.gen_raw_data[p_name] = {}
            if self.generate_mask:
                self.gen_mask[p_name] = {}
                self.gen_nan_raw_data[p_name] = {}
            for seed in range(args['n_seeds']):
                self.rec_info[p_name][seed] = {}
                self.gen_info[p_name][seed] = {}
                self.gen_raw_data[p_name][seed] = {}
                if self.generate_mask:
                    self.gen_mask[p_name][seed] = {}
                    self.gen_nan_raw_data[p_name][seed] = {}

        # Save results
        for res in results:
            p_name = str(res[0]['latent_dim']) + '_' + str(res[0]['hidden_size'])
            self.rec_info[p_name][res[1]] = res[2]
            self.gen_info[p_name][res[1]] = res[3]
            if self.generate_mask:
                self.gen_raw_data[p_name][res[1]] = res[4][0]
                self.gen_mask[p_name][res[1]] = res[4][1]
                self.gen_nan_raw_data[p_name][res[1]] = res[4][2]
            else:
                self.gen_raw_data[p_name][res[1]] = res[4]

    # Since several parameters and seeds are used, we need to save the results in a dictionary
    def set_gauss_results_dictionaries(self, results, args):
        # Create results dictionaries
        for params in args['param_comb']:
            p_name = str(params['latent_dim']) + '_' + str(params['hidden_size'])
            self.gauss_gen_info[p_name] = {}
            self.gauss_gen_raw_data[p_name] = {}
            if self.generate_mask:
                self.gauss_gen_mask[p_name] = {}
                self.gauss_gen_nan_raw_data[p_name] = {}
            for seed in range(args['n_seeds']):
                self.gauss_gen_info[p_name][seed] = {}
                self.gauss_gen_raw_data[p_name][seed] = {}
                if self.generate_mask:
                    self.gauss_gen_mask[p_name][seed] = {}
                    self.gauss_gen_nan_raw_data[p_name][seed] = {}

        # Save results
        for res in results:
            p_name = str(res[0]['latent_dim']) + '_' + str(res[0]['hidden_size'])
            self.gauss_gen_info[p_name][res[1]] = res[5]
            if self.generate_mask:
                self.gauss_gen_raw_data[p_name][res[1]] = res[6][0]
                self.gauss_gen_mask[p_name][res[1]] = res[6][1]
                self.gauss_gen_nan_raw_data[p_name][res[1]] = res[6][2]
            else:
                self.gauss_gen_raw_data[p_name][res[1]] = res[6]

    def postprocess_gen_data(self, gen_info):
        # Denormalize generated samples to check data ranges
        cov_samples = gen_info['cov_samples']
        denorm_gen_df = self.transform_data(cov_samples, denorm=True)

        # Remove negative gaussian samples if the original data didn't have them and convert data types to originals'
        for idx in range(self.columns.size):
            if idx in self.positive_gaussian_cols:
                cov_samples = cov_samples[denorm_gen_df.iloc[:, idx] >= 0]
                gen_info['z'] = gen_info['z'][denorm_gen_df.iloc[:, idx] >= 0]
                gen_info['cov_params'] = gen_info['cov_params'][denorm_gen_df.iloc[:, idx] >= 0]
                gen_info['latent_params'][0] = gen_info['latent_params'][0][denorm_gen_df.iloc[:, idx] >= 0]
                gen_info['latent_params'][1] = gen_info['latent_params'][1][denorm_gen_df.iloc[:, idx] >= 0]
                denorm_gen_df = denorm_gen_df[denorm_gen_df.iloc[:, idx] >= 0]

            # Check if column is gaussian and raw_df has no decimal values
            if self.feat_distributions[idx][0] == 'gaussian':
                no_nan_values = self.processed_df.iloc[:, idx].values[~pd.isnull(self.processed_df.iloc[:, idx].values)]
                if np.amin(np.equal(np.mod(no_nan_values, 1), 0)):
                    # Not necessary to transform to int, since raw_df might be float because of NaNs
                    denorm_gen_df.iloc[:, idx] = denorm_gen_df.iloc[:, idx].round()

        # set imp_norm_df columns names to denorm_gen_df columns names
        denorm_gen_df.columns = self.imp_norm_df.columns
        gen_info['raw_cov_samples'] = denorm_gen_df
        # norm_gen_df = self.transform_data(denorm_gen_df)
        # norm_gen_df.columns = self.imp_norm_df.columns
        gen_info['cov_samples'] = self.transform_data(denorm_gen_df)

        return gen_info

    def save_data_to_csv(self, path, model_path, gen_info):
        self.raw_df.to_csv(path + os.sep + 'raw_data.csv', index=False)
        self.raw_mask.to_csv(path + os.sep + 'mask.csv', index=False)

        # Separate raw_cov_samples into data and mask
        gen_raw_data = gen_info['raw_cov_samples'].iloc[:, :self.processed_df.shape[1]]
        # Before saving data to csv, we need to transform numerical values to categorical if necessary
        # Iter keys from self.str_mapping and transform gen values
        transf_gen_raw_data = gen_raw_data.copy()
        if self.mapping_info is not None:
            for key in self.mapping_info.keys():
                for idx, val in enumerate(self.mapping_info[key]):
                    # If row in transf_gen_raw_data[key] is the same as idx, replace it with val
                    transf_gen_raw_data[key] = transf_gen_raw_data[key].replace(idx, val)
        transf_gen_raw_data.to_csv(path + 'bgm' + os.sep + model_path + os.sep + 'raw_gen_data.csv', index=False)
        gen_data = gen_info['cov_samples'].iloc[:, :self.processed_df.shape[1]]

        if self.generate_mask:
            gen_mask = gen_info['raw_cov_samples'].iloc[:, -self.processed_df.shape[1]:]
            gen_mask = gen_mask.set_axis(self.columns, axis=1)
            gen_mask.to_csv(path + 'bgm' + os.sep + model_path + os.sep + 'gen_mask.csv', index=False)

            # Multiply generated mask to generated data to simulate lost information
            gen_nan_mask = gen_mask.copy()
            gen_nan_mask[gen_nan_mask == 0] = float('nan')
            gen_nan_raw_data = gen_raw_data * gen_nan_mask
            for key in self.mapping_info.keys():
                for idx, val in enumerate(self.mapping_info[key]):
                    # If row in transf_gen_raw_data[key] is the same as idx, replace it with val
                    gen_nan_raw_data[key] = gen_nan_raw_data[key].replace(idx, val)
            gen_nan_raw_data.to_csv(path + 'bgm' + os.sep + model_path + os.sep + 'raw_nan_gen_data.csv', index=False)
            gen_nan_data = gen_data * gen_nan_mask
            return gen_data, gen_mask, gen_nan_data
        else:
            return gen_raw_data

    def save_gauss_data_to_csv(self, path, model_path, gauss_gen_info):
        # Separate raw_cov_samples into data and mask
        gen_raw_data = gauss_gen_info['raw_cov_samples'].iloc[:, :self.processed_df.shape[1]]
        # Before saving data to csv, we need to transform numerical values to categorical if necessary
        # Iter keys from self.str_mapping and transform gen values
        transf_gen_raw_data = gen_raw_data.copy()
        if self.mapping_info is not None:
            for key in self.mapping_info.keys():
                for idx, val in enumerate(self.mapping_info[key]):
                    # If row in transf_gen_raw_data[key] is the same as idx, replace it with val
                    transf_gen_raw_data[key] = transf_gen_raw_data[key].replace(idx, val)
        transf_gen_raw_data.to_csv(path + 'gauss' + os.sep + model_path + os.sep + 'raw_gauss_gen_data.csv', index=False)
        gen_data = gauss_gen_info['cov_samples'].iloc[:, :self.processed_df.shape[1]]

        if self.generate_mask:
            gen_mask = gauss_gen_info['raw_cov_samples'].iloc[:, -self.processed_df.shape[1]:]
            gen_mask = gen_mask.set_axis(self.columns, axis=1)
            gen_mask.to_csv(path + 'gauss' + os.sep + model_path + os.sep + 'gauss_gen_mask.csv', index=False)

            # Multiply generated mask to generated data to simulate lost information
            gen_nan_mask = gen_mask.copy()
            gen_nan_mask[gen_nan_mask == 0] = float('nan')
            gen_nan_raw_data = gen_raw_data * gen_nan_mask
            for key in self.mapping_info.keys():
                for idx, val in enumerate(self.mapping_info[key]):
                    # If row in transf_gen_raw_data[key] is the same as idx, replace it with val
                    gen_nan_raw_data[key] = gen_nan_raw_data[key].replace(idx, val)
            gen_nan_raw_data.to_csv(path + 'gauss' + os.sep + model_path + os.sep + 'raw_nan_gauss_gen_data.csv', index=False)
            gen_nan_data = gen_data * gen_nan_mask
            return gen_data, gen_mask, gen_nan_data
        else:
            return gen_raw_data

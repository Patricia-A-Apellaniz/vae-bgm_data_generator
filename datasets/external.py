"""Train-fitted preprocessing for CSV tables with mixed variable types."""
import numpy as np
import pandas as pd


class TableEncoder:
    def __init__(self, schema=None):
        self.schema = schema or {}
        self.features = []

    def fit(self, frame):
        if frame.empty or not frame.columns.is_unique:
            raise ValueError('The table must have rows and unique column names')
        unknown = set(self.schema) - set(frame.columns)
        if unknown:
            raise ValueError('Schema contains unknown columns: ' + str(sorted(unknown)))
        features = []
        for name in frame:
            values = frame[name].dropna()
            if values.empty:
                raise ValueError('Column has no observed training values: ' + name)
            kind = self.schema.get(name)
            if kind is None:
                kind = 'gaussian' if pd.api.types.is_numeric_dtype(values) else 'categorical'
            if kind not in {'gaussian', 'categorical', 'bernoulli'}:
                raise ValueError('Unsupported distribution for ' + name + ': ' + str(kind))
            feature = {'name': name, 'distribution': kind, 'dtype': str(frame[name].dtype)}
            if kind == 'gaussian':
                numeric = pd.to_numeric(values, errors='raise').astype(float)
                if not np.isfinite(numeric).all():
                    raise ValueError('Non-finite numeric values in ' + name)
                feature.update(mean=float(numeric.mean()), scale=float(numeric.std(ddof=0)) or 1.0,
                               fill=float(numeric.median()))
            else:
                categories = values.unique().tolist()
                if kind == 'bernoulli' and len(categories) != 2:
                    raise ValueError('Bernoulli columns require two training categories: ' + name)
                feature.update(categories=categories, fill=values.mode().tolist()[0])
            features.append(feature)
        self.features = features
        return self

    @property
    def distributions(self):
        return [(f['distribution'], 2 if f['distribution'] == 'gaussian' else
                 len(f['categories']) if f['distribution'] == 'categorical' else 1)
                for f in self.features]

    def transform(self, frame):
        if list(frame.columns) != [f['name'] for f in self.features]:
            raise ValueError('Input columns must match training columns in order')
        columns, masks = [], []
        for feature in self.features:
            values = frame[feature['name']]
            if feature['distribution'] == 'gaussian':
                values = pd.to_numeric(values, errors='raise')
                if np.isinf(values).any():
                    raise ValueError('Infinite values in ' + feature['name'])
                mask = values.notna()
                encoded = (values.fillna(feature['fill']) - feature['mean']) / feature['scale']
            else:
                mapping = {value: i for i, value in enumerate(feature['categories'])}
                encoded = values.map(mapping)
                mask = encoded.notna()
                encoded = encoded.fillna(mapping[feature['fill']])
            columns.append(encoded.to_numpy(dtype=np.float32))
            masks.append(mask.to_numpy(dtype=np.float32))
        return np.column_stack(columns), np.column_stack(masks)

    def inverse_transform(self, matrix):
        result = {}
        for index, feature in enumerate(self.features):
            values = matrix[:, index]
            if feature['distribution'] == 'gaussian':
                values = values * feature['scale'] + feature['mean']
                if feature['dtype'].startswith(('int', 'uint', 'Int', 'UInt')):
                    values = np.rint(values)
            else:
                indices = np.rint(values).astype(int)
                if np.any(indices < 0) or np.any(indices >= len(feature['categories'])):
                    raise ValueError('Generated category outside training vocabulary')
                values = np.asarray(feature['categories'])[indices]
            result[feature['name']] = values
        return pd.DataFrame(result)

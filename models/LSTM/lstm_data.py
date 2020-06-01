from ..utils import dataloader as loader
import numpy as np

class TimeFeature:
    def __init__(self, name, series, *kwargs):
        self.series = series
        self.name = name
        if 'norm' in kwargs:
            self.norm = kwargs['norm']
        else:
            self.norm = False
        if 'diff_order' in kwargs:
            self.diff_order = kwargs['diff_order']
            self.initial_cond = []
        else:
            self.diff_order = 0
            self.initial_cond = []
        if 'target' in kwargs:
            self.target = kwargs['target']
        else:
            self.target = True

    def __repr__(self):
        return f'[ name: {self.name}, norm: {self.norm}, ' + \
        f'diff: {self.diff_order}, target: {self.target} ]'

class TimeIndependentFeature:
    def __init__(self, name, series):
        self.name = name
        self.series = series

    def __getitem__(self, index):
        return self.series[index]

    def __repr__(self):
        return f'Feature: {self.name}'

class Data:
    def __init__(self, data_format, loader_args={}):
        self.datadict = loader.load_covid_timeseries(**loader_args)
        self.demographics = loader.load_demographics_data()

        self.data_format(data_format)

    def data_format(self, config):
        '''
        time_features
            - name
            - norm
            - diff_order
            - target
        '''

        if 'time_features' not in config:
            raise ValueError('Must include time features') 
        self.time_features = []
        for feature in config['time_features']:
            series = None

            # Get time feature by name
            if 'name' not in feature:
                raise ValueError('Must specify name of time feature')
            if feature['name'] in self.datadict.keys():
                series = self.datadict[feature['name']]
            #elif in 'mobility', etc.
            else:
                raise ValueError(f'{feature["name"]} is not a recognized ' +
                    'time feature name')
            feature_obj = TimeFeature(feature['name'], series)

            if 'target' in feature:
                feature_obj.target = feature['target']

            # Normalize by population?
            if 'norm' in feature and feature['norm']:
                series = self.norm(series, feature['norm'])
                feature_obj.norm = feature['norm'] 

            # Difference
            if 'diff_order' in feature:
                diff_order = feature['diff_order']
                differenced, initial_cond = self.difference(series, diff_order)
                feature_obj.series = differenced
                feature_obj.diff_order = diff_order
                feature_obj.initial_cond = initial_cond

            self.time_features.append(feature_obj)

        self.n_counties = self.time_features[0].series.shape[0]
        for feature in self.time_features:
            if feature.series.shape[0] != self.n_counties:
                raise ValueError('Every time feature must have the same number' +
                    ' of counties')
        self.targets = [feature for feature in self.time_features if 
                        feature.target]
        self.n_targets = len(self.targets)

        self.time_independent_features = []
        if 'time_independent_features' in config:
            for feature_name in config['time_independent_features']:
                if feature_name in self.demographics.columns:
                    self.time_independent_features.append(
                        TimeIndependentFeature(feature_name,
                            self.demographics[feature_name]))
                else:
                    raise ValueError(f'{feautre_name} is not a recognized ' +
                    'time independent feature name')

        if 'time_context' in config:
            self.time_context = config['time_context']
        else:
            self.time_context = False

        self.n_time_features = len(self.time_features)
        self.n_features = len(self.time_features) + len(
            self.time_independent_features) + int(self.time_context)
        self.tsteps = self.series_rep().shape[1]

    def norm(self, series, val):
        pop = np.expand_dims(self.demographics['total_pop'].values, axis=1)
        series = series / pop * val
        return series

    def unnorm(self, series, val):
        pop = np.expand_dims(self.demographics['total_pop'].values, axis=1)
        series = series * pop / val
        return series

    def difference(self, series, order):
        if order >= series.shape[1]:
            raise ValueError('cannot difference that many times')
        initial_conditions = []
        for i in range(order):
            d0 = np.expand_dims(series[:, 0], axis=1)
            initial_conditions.append(d0)
            series = np.diff(series)

        return series, initial_conditions

    def undifference(self, series, initial_conditions, order, axis=1):
        if len(initial_conditions) != order:
            print(len(initial_conditions), order)
            raise ValueError('Invalid initial conditions')

        for i in range(order):
            d0 = initial_conditions[-1]
            series = np.hstack([d0, series])
            series = np.cumsum(series, axis=axis)
            initial_conditions = initial_conditions[:-1]

        return series

    def series_rep(self):
        return self.time_features[0].series

    def get_training_data(self, lag=7, k=7, val_steps=9, dense=False):
        '''
        Assumes calibrated time series

        lag - number of lag features to include
        k - time steps to predict
        val_steps - number of steps from the end to save for validation
        '''
        if dense:
            k = 1

        X_train = []
        y_train = []
        X_test = []
        y_test = []

        for county in range(self.n_counties):
            # Skip all nan counties
            if any([np.all(np.isnan(feature.series)) for feature 
                in self.time_features]):
                continue

            ### Assumption: all time series have been calibrated together, and
            ### nans are at the end. (Result of calibration in dataloader)

            f = self.series_rep()[county]
            # index of first nan for that county
            if np.all(~np.isnan(f)):
                s = f.shape[0]
            else: 
                s = np.argmax(np.isnan(f))
            # t + lag is the prediction horizon (i.e. f[t + lag] is the first
            # step to predict, with lag context from f[t])
            for t in range(s):
                if np.any(np.isnan(f[t:t+lag+k])) or t + lag + k > s:
                    break

                x = np.vstack([feature.series[county, t:t + lag] for feature 
                    in self.time_features])
                x = np.vstack([x] + [np.full((lag, ), feature[county])
                    for feature in self.time_independent_features])
                if self.time_context:
                    x = np.vstack([x, np.full((lag, ), t)])
                x = x.T

                if dense:
                    y = np.vstack([feature.series[county, t + 1:t + lag + 1]
                    for feature in self.targets])
                else:
                    y = np.vstack([feature.series[county, t + lag:t + lag + k]
                    for feature in self.targets])
                y = y.T

                # t + lag + k <= s - v for training
                if t <= s - val_steps - lag - k:
                    X_train.append(x)
                    y_train.append(y)
                else:
                    X_test.append(x)
                    y_test.append(y)

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        return X_train, y_train, X_test, y_test

    def original(self):
        undifferenced = []

        for feature in self.time_features:
            undiff = self.undifference(feature.series, feature.initial_cond,
                feature.diff_order)
            if feature.norm:
                undiff = self.unnorm(undiff, feature.norm)

            undifferenced.append(undiff)

        return undifferenced

    def original_with_predictions(self, prediction, t):
        undifferenced = []
        for i, feature in enumerate(self.time_features):
            combined_feature = np.copy(feature.series)
            horizon = prediction.shape[1]
            combined_feature = np.concatenate([combined_feature,
                np.full((self.n_counties, horizon), np.nan)], axis=1)
            for county in range(self.n_counties):
                f = combined_feature[county]
                s = np.argmax(np.isnan(f))

                if np.all(np.isnan(f)):
                    continue
                if s - t < 0:
                    continue

                combined_feature[county, (s - t):(s - t + horizon)] = \
                    prediction[county, :, i]

            undiff = self.undifference(combined_feature, feature.initial_cond,
                feature.diff_order)

            if feature.norm:
                undiff = self.unnorm(undiff, feature.norm)

            undifferenced.append(undiff)

        return undifferenced
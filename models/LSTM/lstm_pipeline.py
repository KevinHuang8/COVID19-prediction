import numpy as np
#import tensorflow as tf

from . import lstm_data as dt
from . import lstm_models as models
from ..utils import dataloader as loader

class Pipeline:
    def __init__(self, data_format, model_params, training_params, horizon, 
        loader_args={}):
        '''
        data_format,
        model_params
            - name
            - lag_features
            - prediction_horizon
            - arguments passed into model function
                - quantiles
                - loss
                - optimizer
        training_params
            - val_steps
            - predict_time (when to start making predictions, in time from end)
            - epochs
            - batch_size
            - dense

        horizon: total horizon predictions needed for
        '''
        
        self.data = dt.Data(data_format, loader_args=loader_args)

        try:
            self.lag = model_params['lag_features']
            self.k = model_params['prediction_horizon']
        except KeyError:
            raise ValueError('Musty specify lag features and prediction ' +
                'horizon in model specification')
        try:
            val_steps = training_params['val_steps']
        except KeyError:
            raise ValueError('Must specify validation steps in training params')

        if 'dense' in training_params:
            self.dense = training_params['dense']
        else:
            self.dense = False

        self.X_train, self.y_train, self.X_test, self.y_test = \
            self.data.get_training_data(lag=self.lag, k=self.k, 
                val_steps=val_steps, dense=self.dense)

        try:
            name = model_params['name']
        except KeyError:
            raise ValueError('Must specify model name')

        try:
            model_creator = getattr(models, name)
        except AttributeError:
            raise ValueError(f'{name} is not a recognized model name')

        self.model_params = model_params
        model_params = {key: model_params[key] for key in model_params if key 
        not in ['name', 'lag_features', 'prediction_horizon']}

        self.model = model_creator(self.X_train, self.y_train, **model_params)

        if 'quantiles' in self.model_params and self.model_params['quantiles']:
            self.quantiles = True
            self.n_quantiles = self.model_params['quantiles']
        else:
            self.quantiles = False        

        if 'epochs' not in training_params:
            raise ValueError('Must specify epochs in training params')
        if 'batch_size' not in training_params:
            raise ValueError('Must specify batch size in training params')

        self.training_params = training_params
        self.horizon = horizon

    def run(self):
        training_params = {key: self.training_params[key] for key in 
            self.training_params if key not in ['val_steps', 'predict_time',
            'dense']}
        
        self.model.fit(self.X_train, self.y_train, 
            validation_data=(self.X_test, self.y_test), shuffle=True,
            **training_params)

        if 'predict_time' not in self.training_params:
            self.training_params['predict_time'] = 0
        
        self.t = self.lag + self.training_params['predict_time']

        self.predictions = self.predict(t=self.t, dense=self.dense)

    def predict(self, t=0, dense=False):
        if t < self.lag:
            raise ValueError('Cannot predict from point less than number of lags')
        if self.quantiles:
            predictions = np.zeros((self.n_quantiles, self.data.n_counties, 
                self.horizon, self.data.n_targets))
        else:
            predictions = np.zeros((self.data.n_counties, 
                self.horizon, self.data.n_targets))

        combined_features = np.full((self.data.n_time_features, 
            self.data.n_counties, self.data.tsteps), np.nan)
        for i, feature in enumerate(self.data.time_features):
            combined_features[i, :, :] = feature.series
        combined_features = np.concatenate([combined_features, 
            np.full((self.data.n_time_features, self.data.n_counties, 
                self.horizon), np.nan)], axis=2)

        # for county in range(self.data.n_counties):
        #     f = combined_features[0, county]
        #     if np.all(np.isnan(f)):
        #         continue

        #     # s is the end of the series
        #     s = np.argmax(np.isnan(f))

        #     combined_features[:, (s - t + self.lag):] = np.nan

        for j in range(self.horizon // self.k):
            X = np.zeros((self.data.n_counties, self.lag, self.data.n_features))

            for county in range(self.data.n_counties):
                f = combined_features[0, county]
                if np.all(np.isnan(f)):
                    continue

                # s is the end of the series
                s = np.argmax(np.isnan(f))

                #i = s - self.lag
                i = s - t
                if i < 0:
                    continue

                x = combined_features[:, county, i:i + self.lag]
                x = np.vstack([x] + [np.full((self.lag, ), feature[county])
                    for feature in self.data.time_independent_features])
                if self.data.time_context:
                    x = np.vstack([x, np.full((self.lag, ), i)])
                x = x.T

                X[county, :] = x

            y_pred = self.model.predict(X)
            y_pred = np.array(y_pred)

            if dense:
                if self.quantiles:
                    y_pred = y_pred[:, :, -1:, :]
                else:
                    y_pred = y_pred[:, -1:, :]


            if self.quantiles:
                # y_pred = y_pred.reshape(self.n_quantiles,
                # self.data.n_counties, self.k, self.data.n_targets)
                predictions[:, :, (j*self.k):((j + 1)*self.k), :] = y_pred
            else:
                # y_pred = y_pred.reshape(
                # self.data.n_counties, self.k, self.data.n_targets)
                predictions[:, (j*self.k):((j + 1)*self.k), :] = y_pred

            for county in range(self.data.n_counties):
                f = combined_features[0, county]
                if np.all(np.isnan(f)):
                    continue

                # s is the end of the series
                s = np.argmax(np.isnan(f))
                i = s

                if self.quantiles:
                    combined_features[:, county, i:i + self.k] = \
                    np.moveaxis(y_pred, [1, 2, 3], [2, 3, 1])[4, :, county, :]
                else:
                    combined_features[:, county, i:i + self.k] = \
                    np.moveaxis(y_pred, [0, 1, 2], [1, 2, 0])[:, county, :]

        return predictions

    def get_predictions(self):
        if self.quantiles:
            quantile_predictions = []
            for q in range(self.n_quantiles):
                p = self.data.original_with_predictions(self.predictions[q], 
                    self.training_params['predict_time'])
                quantile_predictions.append(p)
            return quantile_predictions
        else:
            return self.data.original_with_predictions(self.predictions, 
                self.training_params['predict_time'])
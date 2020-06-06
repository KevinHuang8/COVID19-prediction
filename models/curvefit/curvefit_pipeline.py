import csv
import pickle
import datetime as dt
import numpy as np
import pandas as pd
from . import curvefit_models as models
from ..utils import dataloader as loader

class Data:
    '''
    Class that stores and processes the data.
    '''
    def __init__(self, data_format):
        '''
        data_format - a dict with the following entries
            - 'name': cumulative data to load using 
              dataloader.load_covid_timeseries()
                - either 'cases_raw' or 'deaths_raw'
            - 'smoothing': size of moving average to smooth timeseries
            - 'val_steps': number of data points from end to withold for 
              validation
            - international: international county to use
        '''
        try:
            country = data_format['international']
        except KeyError:
            datadict = loader.load_covid_timeseries()

            name = data_format['name']
            self.raw_data = datadict[name]
        else:
            self.raw_data = loader.load_international_data(country)
        
        self.n_counties = self.raw_data.shape[0]

        smoothing = data_format['smoothing']
        self.val_steps = data_format['val_steps']

        # Pre-processing steps:
        #   - only take cumulative data > 0
        #   - smooth
        #   - exclude counties with not enough >0 datapoints
        self.cumulative_series = {}
        self.daily_change = {}
        self.daily_smoothed = {}
        for county, series in enumerate(self.raw_data):
            if series[series > 0].shape[0] < self.val_steps + 2:
                continue
            self.cumulative_series[county] = series[series > 0]
            self.daily_change[county] = np.diff(self.cumulative_series[county])
            self.daily_smoothed[county] = loader.smooth_timeseries(
                self.daily_change[county], smoothing, axis=0)

    def get_training_data(self, county, cumulative=False):
        '''
        county - a county index
        cumulative - whether to use cumulative or daily data

        Return a train/test split for the county
        '''
        if cumulative:
            series = self.cumulative_series[county]
        else:
            series = self.daily_smoothed[county]

        end = series.shape[0]
        split = end - self.val_steps

        X_train = np.arange(split)
        y_train = series[:split]

        X_test = np.arange(split, end)
        y_test = series[split:]

        return X_train, y_train, X_test, y_test

class Pipeline:
    '''
    Encapsulates the entire curvefit model training process.
    '''
    def __init__(self, data_format, model_params, horizon, use_cumulative=None):
        '''
        data_format - see Data
        model_params - a dict with the following elements:
            - 'name': curvefit model to use [required]
            - 'params': parameters to pass into model constructor
            - 'use_gp': 
                - 'all' means to use GP step on all counties
                - None means to skip GP step for all counties
                - a list of county indices to apply GP step to
            - 'gp_params': parameters to pass into GP model, see curvefit_models
        horizon - number of steps to predict, integer
        use_cumulative - a dict that maps county indices to a boolean,
        determining whether each county should use a cumulative model or not.
        If not provided, it is automatically determined based on validation
        set performance.
        '''
        self.data = Data(data_format)
        self.data_format = data_format

        try:
            name = model_params['name']
        except KeyError:
            raise ValueError('Must specify model name')
        try:
            model_creator = getattr(models, name)
        except AttributeError:
            raise ValueError(f'{name} is not a valid model name')
        gp_model = getattr(models, 'GPModel')
        try:
            params = model_params['params']
        except KeyError:
            params = {}
        try:
            self.use_gp = model_params['use_gp']
            if self.use_gp is None:
                self.use_gp = []
        except KeyError:
            self.use_gp = []
        try:
            self.gp_params = model_params['gp_params']
        except KeyError:
            self.gp_params = {}

        self.horizon = horizon

        # We have a separate model for each county
        self.models = {}
        for county in range(self.data.n_counties):
            try:
                data_max = self.data.cumulative_series[county].max()
            except KeyError:
                continue
            if self.use_gp == 'all' or county in self.use_gp:
                model = gp_model(data_max=data_max, *params)
            else:
                model = model_creator(data_max=data_max, *params)
            self.models[county] = model

        if use_cumulative:
            self.use_cumulative = use_cumulative
        else:
            self.use_cumulative = None

    def run(self):
        '''
        Train each model.

        If use_cumulative is not determined yet, we need to run both models
        for each county and determine which one is better, based on validation
        set performance. 
        '''
        if not self.use_cumulative:
            self.blind_run()
        else:
            self.warm_run()

    def blind_run(self):
        '''
        Train each county with a cumulative and non-cumulative model, keeping
        the better one. 

        Note: We cannot use predict() or get_combined_predictions() 
        if doing a blind run. One blind run should be done to determine
        use_cumulative, and then another run should be made for the actual
        predictions.
        '''
        self.predictions = {}
        self.use_cumulative = {}
        for county in range(self.data.n_counties):
            print(f'Fitting {county}/{self.data.n_counties - 1}', end='\r')
            try:
                model = self.models[county]
            except KeyError:
                shape = (self.horizon, )
                s = loader.smooth_timeseries(
                    np.diff(self.data.raw_data[county]), axis=0)
                self.predictions[county] = np.full(shape, 
                    s[-1])
                continue
            
            ### First train on differenced data
            X_train1, y_train1, X_test1, y_test1 = \
                self.data.get_training_data(county, cumulative = False)
            model.set_cumulative(False)
            model.fit(X_train1, y_train1)

            y_pred1 = model.predict(X_test1)

            series = self.data.daily_smoothed[county]
            end = series.shape[0]
            pred_x = np.arange(end - self.data.val_steps, 
                end - self.data.val_steps + self.horizon)
            prediction1 = model.predict(pred_x)

            ### Then train on cumulative data
            X_train2, y_train2, X_test2, y_test2 = \
                self.data.get_training_data(county, cumulative = True)
            model.set_cumulative(True)
            model.fit(X_train2, y_train2)

            series = self.data.cumulative_series[county]
            end = series.shape[0]
            x = np.concatenate([[X_test2[0] - 1], X_test2])
            y_pred2 = model.predict(x)

            pred_x = np.arange(end - self.data.val_steps, 
                end - self.data.val_steps + self.horizon + 1)
            prediction2 = model.predict(pred_x)

            ### Then compare the two on the validation data
            ### and choose the better method
            err1 = np.sum((y_test1 - y_pred1)**2)
            err2 = np.sum((y_test1 - y_pred2)**2)

            if err1 < err2:
                self.use_cumulative[county] = False
                self.predictions[county] = prediction1
            else:
                self.use_cumulative[county] = True
                self.predictions[county] = prediction2

    def warm_run(self):
        '''
        Train each model, with use_cumulative already determined. 
        '''
        for county in range(self.data.n_counties):
            print(f'Fitting {county}/{self.data.n_counties - 1}', end='\r')
            try:
                model = self.models[county]
            except KeyError:
                continue

            try:
                use_cumulative = self.use_cumulative[county]
            except KeyError:
                continue

            if use_cumulative:
                X_train, y_train, X_test, y_test = \
                    self.data.get_training_data(county, cumulative = True)
                model.set_cumulative(True)
                unsmoothed = self.data.cumulative_series[county][X_train]
                unsmoothed = np.diff(unsmoothed)
            else:
                X_train, y_train, X_test, y_test = \
                    self.data.get_training_data(county, cumulative = False)
                model.set_cumulative(False)
                unsmoothed = self.data.daily_change[county][X_train]

            if self.use_gp == 'all' or county in self.use_gp:
                model.fit(X_train, y_train, unsmoothed, **self.gp_params)
            else:
                model.fit(X_train, y_train)                

        self.predict()

    def predict(self, quantiles=False, samples=100):
        '''
        quantiles -  either False, or a list of quantiles to predict for
        samples - number of samples to take when determining quantiles
        (does not apply to GP step)

        Predict into the future, either a single value or quantiles as
        specified.

        Results stored in self.predictions, which is a dict that maps a
        county index to a 2D np array. Axis 0 is time, and axis 1 is quantiles.
        '''
        self.predictions = {}
        print('')
        for county in range(self.data.n_counties):
            print(f'Predicting {county}/{self.data.n_counties - 1}', end='\r')
            # For any counties with not enough data (no model), then
            # we simply predict a constant value, based on the last
            # value known.
            try:
                model = self.models[county]
            except KeyError:
                if quantiles:
                    shape = (self.horizon, len(quantiles))
                else:
                    shape = (self.horizon, )
                s = loader.smooth_timeseries(
                    np.diff(self.data.raw_data[county]), axis=0)
                self.predictions[county] = np.full(shape, 
                    s[-(1 + self.data.val_steps)])
                continue

            try:
                use_cumulative = self.use_cumulative[county]
            except KeyError:
                if quantiles:
                    shape = (self.horizon, len(quantiles))
                else:
                    shape = (self.horizon, )
                s = loader.smooth_timeseries(
                    np.diff(self.data.raw_data[county]), axis=0)
                self.predictions[county] = np.full(shape, 
                    s[-(1 + self.data.val_steps)])
                continue

            if use_cumulative:
                series = self.data.cumulative_series[county]
                end = series.shape[0]
                # cumulative models need an extra step at the end, because
                # the data is differenced
                x = np.arange(end - self.data.val_steps, 
                    end - self.data.val_steps + self.horizon + 1)
            else:
                series = self.data.daily_smoothed[county]
                end = series.shape[0]
                x = np.arange(end - self.data.val_steps, 
                    end - self.data.val_steps + self.horizon)

            if quantiles:
                y_pred = model.predict_quantiles(x, quantiles, samples)
            else:
                y_pred = model.predict(x)
            self.predictions[county] = y_pred

    def get_combined_predictions(self, quantiles=False, samples=100):
        '''
        Computes predictions, but concatenates the future predicted values
        with the past time series. This is mainly used for visualization
        purposes.
        '''
        ## note: rn, only works correctly w/ warm runs, otherwise models aren't
        ## correct versions
        combined = {}
        for county in range(self.data.n_counties):
            print(f'Predicting {county}/{self.data.n_counties - 1}', end='\r')
            n = self.data.raw_data[county].shape[0]

            try:
                model = self.models[county]
            except KeyError:
                if quantiles:
                    shape = (n, len(quantiles))
                else:
                    shape = (n, )
                s = loader.smooth_timeseries(
                    np.diff(self.data.raw_data[county]), axis=0)
                combined[county] = np.full(shape, 
                    s[-1])
                continue

            try:
                use_cumulative = self.use_cumulative[county]
            except KeyError:
                if quantiles:
                    shape = (n, len(quantiles))
                else:
                    shape = (n, )
                s = loader.smooth_timeseries(
                    np.diff(self.data.raw_data[county]), axis=0)
                combined[county] = np.full(shape, 
                    s[-1])
                continue

            if use_cumulative:
                series = self.data.cumulative_series[county]
                end = series.shape[0]
                x = np.arange(0, end - self.data.val_steps + self.horizon + 1)
            else:
                model.set_cumulative(False)
                series = self.data.daily_smoothed[county]
                end = series.shape[0]
                x = np.arange(0, end - self.data.val_steps + self.horizon)

            if quantiles:
                y_pred = model.predict_quantiles(x, quantiles, samples)
            else:
                y_pred = model.predict(x)
            combined[county] = y_pred
        self.combined = combined
        return combined

    def write_to_file(self, filename, sample_dir, quantiles):
        '''
        filename - place to save predictions to, in csv format
        sample_dir - the sample submission file. This will match all rows
        present in the sample submission
        quantiles - list of quantiles to report
            - this must match quantiles used for prediction

        Creates the submission file. Must have run .predict() before writing
        to file.
        '''
        # Ensure that the created file has the same counties as the sample
        df = pd.read_csv(sample_dir)
        info = loader.load_info_raw()

        i = df.set_index('id').sort_index().index
        # sub_fips - all FIPS in the sample
        sub_fips = np.unique([s[11:] for s in i])
        # data_fips - all FIPS predicted
        data_fips = [s.lstrip('0') for s in info['FIPS']]

        # Get all FIPS that we have predicted for, but are not in the sample
        dont_include = []
        for fips in data_fips:
            if fips not in sub_fips:
                dont_include.append(fips)

        # Get all FIPS that are in the sample, but we haven't predicted for
        must_include = []
        for fips in sub_fips:
            if fips not in data_fips:
                must_include.append(fips)

        start_date = '04/01/2020'
        end_date = '06/30/2020'
        # %Y or %y randomly work sometimes for some reason
        try:
            predict_start = dt.datetime.strptime(info.columns[-1], '%m/%d/%Y') \
                - dt.timedelta(days=self.data.val_steps) + dt.timedelta(days=1)
        except ValueError:
            predict_start = dt.datetime.strptime(info.columns[-1], '%m/%d/%y') \
                - dt.timedelta(days=self.data.val_steps) + dt.timedelta(days=1)

        predictions = self.predictions

        # Write predictions to file, making sure to include must_include and
        # excluding counties in dont_include
        to_write = [['id'] + [str(q) for q in quantiles]]
        for county in predictions:
            print(f'writing {county}/{len(predictions) - 1}', end='\r')
            
            fips = info.iloc[county]['FIPS'].lstrip('0')
            if fips in dont_include:
                continue
            
            county_pred = predictions[county]
            
            date = dt.datetime.strptime(start_date, '%m/%d/%Y')
            end = dt.datetime.strptime(end_date, '%m/%d/%Y')
            predict_end = predict_start + dt.timedelta(days=self.horizon - 1)
            while date <= end:
                id_ = date.strftime('%Y-%m-%d-') + fips
                
                if predict_start <= date <= predict_end and (
                    fips not in ['36005', '36047', '36081', '36085']):
                    index = (date - predict_start).days
                    p = list(county_pred[index, :])
                else:
                    p = [0 for i in range(len(quantiles))]
                    
                to_write.append([id_] + p)
                date = date + dt.timedelta(days=1)

        for fips in must_include:
            date = dt.datetime.strptime(start_date, '%m/%d/%Y')
            end = dt.datetime.strptime(end_date, '%m/%d/%Y')
            while date <= end:
                id_ = date.strftime('%Y-%m-%d-') + fips
                p = [0 for i in range(len(quantiles))]
                    
                to_write.append([id_] + p)
                date = date + dt.timedelta(days=1)

        with open(filename, "w+", newline='') as f:
                csv_writer = csv.writer(f, delimiter = ",")
                csv_writer.writerows(to_write)

    def save(self, filename):
        '''
        Save the pipeline to filename.
        '''
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as file:
            saved = pickle.load(file)
        return saved

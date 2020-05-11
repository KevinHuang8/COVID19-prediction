import csv
import datetime as dt
import numpy as np
import pandas as pd
import curvefit_models as models
import dataloader as loader

class Data:
    def __init__(self, data_format):
        datadict = loader.load_covid_timeseries()

        name = data_format['name']
        self.raw_data = datadict[name]

        self.n_counties = self.raw_data.shape[0]

        smoothing = data_format['smoothing']
        self.val_steps = data_format['val_steps']

        self.cumulative_series = {}
        self.daily_change = {}
        self.daily_smoothed = {}
        for county, series in enumerate(self.raw_data):
            if series[series > 0].shape[0] < self.val_steps + 2:
                continue
            self.cumulative_series[county] = series[series > 0]
            self.daily_change[county] = np.diff(self.cumulative_series[county])
            self.daily_smoothed[county] = self.smooth_timeseries(
                self.daily_change[county], smoothing)

    def smooth_timeseries(self, t, size=5):
        '''Smooth the function by taking a moving average of "size" time steps'''
        average_filter = np.full((size, ), 1 / size)

        t = np.pad(t, [(size // 2, size // 2)], mode='edge')
        return np.apply_along_axis(lambda r: np.convolve(r, average_filter, 
            mode='valid'), axis=0, arr=t)

    def get_training_data(self, county, cumulative=False):
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
    def __init__(self, data_format, model_params, horizon, use_cumulative=None):

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
        try:
            params = model_params['params']
        except KeyError:
            params = {}

        self.horizon = horizon
        self.predict_time = self.data.val_steps

        self.models = {}
        for county in range(self.data.n_counties):
            try:
                data_max = self.data.cumulative_series[county].max()
            except KeyError:
                continue
            model = model_creator(data_max=data_max, *params)
            self.models[county] = model

        if use_cumulative:
            self.use_cumulative = use_cumulative
        else:
            self.use_cumulative = None

    def run(self):
        if not self.use_cumulative:
            self.blind_run()
        else:
            self.warm_run()

    def blind_run(self):
        self.predictions = {}
        self.use_cumulative = {}
        for county in range(self.data.n_counties):
            print(f'Fitting {county}/{self.data.n_counties - 1}', end='\r')
            try:
                model = self.models[county]
            except KeyError:
                self.predictions[county] = np.full((self.horizon, ), 
                    self.data.raw_data[county, -1])
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
                try:
                    model.fit(X_train, y_train)
                except ValueError as e:
                    print(county)
                    raise e
            else:
                X_train, y_train, X_test, y_test = \
                    self.data.get_training_data(county, cumulative = False)
                model.set_cumulative(False)
                model.fit(X_train, y_train)                

        self.predict()

    def predict(self, quantiles=False, samples=100):
        self.predictions = {}
        print('')
        for county in range(self.data.n_counties):
            print(f'Predicting {county}/{self.data.n_counties - 1}', end='\r')
            try:
                model = self.models[county]
            except KeyError:
                if quantiles:
                    shape = (self.horizon, len(quantiles))
                else:
                    shape = (self.horizon, )
                self.predictions[county] = np.full(shape, 
                    self.data.raw_data[county, -1])
                continue

            try:
                use_cumulative = self.use_cumulative[county]
            except KeyError:
                if quantiles:
                    shape = (self.horizon, len(quantiles))
                else:
                    shape = (self.horizon, )
                self.predictions[county] = np.full(shape, 
                    self.data.raw_data[county, -1])
                continue

            if use_cumulative:
                series = self.data.cumulative_series[county]
                end = series.shape[0]
                x = np.arange(end - self.data.val_steps, 
                    end - self.data.val_steps + self.horizon + 1)
            else:
                series = self.data.daily_smoothed[county]
                end = series.shape[0]
                x = np.arange(end - self.data.val_steps, 
                    end - self.data.val_steps + self.horizon)

            if quantiles:
                y_pred = model.predict_quantiles(x, quantiles, samples, county)
            else:
                y_pred = model.predict(x)
            self.predictions[county] = y_pred

    def get_combined_predictions(self, quantiles=False, samples=100):
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
                combined[county] = np.full(shape, 
                    self.data.raw_data[county, -1])
                continue

            try:
                use_cumulative = self.use_cumulative[county]
            except KeyError:
                if quantiles:
                    shape = (n, len(quantiles))
                else:
                    shape = (n, )
                combined[county] = np.full(shape, 
                    self.data.raw_data[county, -1])
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
                y_pred = model.predict_quantiles(x, quantiles, samples, county)
            else:
                y_pred = model.predict(x)
            combined[county] = y_pred
        return combined

    def write_to_file(self, filename, sample_dir, quantiles):
        df = pd.read_csv(sample_dir)
        info = loader.load_info_raw()

        i = df.set_index('id').sort_index().index
        sub_fips = np.unique([s[11:] for s in i])
        data_fips = [s.lstrip('0') for s in info['FIPS']]

        dont_include = []
        for fips in data_fips:
            if fips not in sub_fips:
                dont_include.append(fips)

        must_include = []
        for fips in sub_fips:
            if fips not in data_fips:
                must_include.append(fips)

        start_date = '04/01/2020'
        end_date = '06/30/2020'
        predict_start = dt.datetime.strptime(info.columns[-1], '%m/%d/%y')
        predict_start = predict_start + dt.timedelta(days=1)

        predictions = self.predictions

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
                
                if predict_start <= date <= predict_end:
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
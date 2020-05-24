import numpy as np
import scipy
from scipy.interpolate import interp1d
from scipy.stats import norm

import dataloader as loader
from gp_model import GPCasesDeathsModel
from clustering import cluster_counties

class Pipeline:
    def __init__(self, cluster_params, gp_params, smoothing=7, min_delay=7,
            val_steps=0, quantiles=None):
        datadict = loader.load_covid_timeseries(cases_cutoff=0)
        self.cases_all = datadict['cases_raw']
        self.deaths_all = datadict['deaths_raw']

        self.min_delay = min_delay
        self.smoothing = smoothing
        self.val_steps = 0
        self.gp_params = gp_params
        if quantiles is None:
            self.quantiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        else:
            self.quantiles = quantiles
        print('Clustering...', end='')
        try:
            self.clusters = loader.load_from_otherdata('clusters.dat')
            self.cluster_ids = loader.load_from_otherdata('cluster_id.dat')
        except FileNotFoundError:
            self.clusters, self.cluster_ids = cluster_counties(**cluster_params)
        print('...done clustering.')

        self.get_cluster_data()
        self.calc_best_delay()
        self.get_training_data()

    def get_cluster_data(self):    
        self.cases_raw = {}
        self.deaths_raw = {}
        self.cases = {}
        self.deaths = {}
        self.cases_smooth = {}
        self.deaths_smooth = {}

        for i, cluster in enumerate(self.clusters):
            print(f'Processing data for cluster {i + 1}/{len(self.clusters)}',
                end='\r')
            self.cases_raw[i] = np.array([self.cases_all[county] 
                for county in cluster])
            self.deaths_raw[i] = np.array([self.deaths_all[county] 
                for county in cluster])
            
            self.cases[i] = np.diff(self.cases_raw[i], axis=1)
            self.deaths[i] = np.diff(self.deaths_raw[i], axis=1)

            self.cases_smooth[i] = loader.smooth_timeseries(self.cases[i], 
                self.smoothing)
            self.deaths_smooth[i] = loader.smooth_timeseries(self.deaths[i], 
                self.smoothing)
        print()

    def calc_best_delay(self):
        self.best_delays = {}

        delays = np.arange(1, 15)
        for c in range(len(self.clusters)):
            print(f'Calculating best delay for cluster ' + \
             f'{c + 1}/{len(self.clusters)}', end='\r')
            corr = []
            for delay in delays:
                cases_past = []
                deaths_curr = []
                for county in range(self.cases_smooth[c].shape[0]):
                    for i in range(delay, 
                        self.cases_smooth[c].shape[1] - self.val_steps):
                        if i < 0.5 * self.cases_smooth[c].shape[1]:
                            continue
                        if self.deaths[c][county, i - delay] == 0:
                            continue
                        cases_past.append(
                            self.cases_smooth[c][county, i - delay])
                        deaths_curr.append(self.deaths_smooth[c][county, i])
                co = np.corrcoef(cases_past, deaths_curr)[0, 1]
                corr.append(co)

            best_delays = np.argsort(corr)[::-1] + 1
            best_delay = best_delays[best_delays >= self.min_delay][0]

            self.best_delays[c] = best_delay
        print()

    def get_training_data(self):
        self.cases_past = {}
        self.deaths_curr = {}

        for c in range(len(self.clusters)):
            print(f'Calculating training data for cluster ' +
                f'{c + 1}/{len(self.clusters)}', end='\r')
            delay = self.best_delays[c]
            cases_past = []
            deaths_curr = []
            for county in range(self.cases_smooth[c].shape[0]):
                for i in range(delay + 1, 
                    self.cases_smooth[c].shape[1] - self.val_steps):
                    if i < 0.5 * self.cases_smooth[c].shape[1]:
                            continue
                    if self.deaths[c][county, i - delay] == 0:
                        continue
                    cases_past.append(self.cases_smooth[c][county, i - delay])
                    deaths_curr.append(self.deaths_smooth[c][county, i])
            self.cases_past[c] = cases_past
            self.deaths_curr[c] = deaths_curr
        print()

    def run(self, start=0):
        self.models = {}
        for c in range(start, len(self.clusters)):
            print(f'fitting cluster {c + 1}/{len(self.clusters)}')
            self.models[c] = GPCasesDeathsModel(**self.gp_params)
            self.models[c].fit(self.cases_past[c], self.deaths_curr[c])
            self.models[c].save(f'{c}.dat', 'gp_cluster')
        print()

    def interpolate_percentiles(self, point, quantile_values, reverse=True):
        y = quantile_values
        x = self.quantiles
        if reverse:
            f = interp1d(y, x)
        else:
            f = interp1d(x, y)
        return f(point)

    def get_bias(self, county, c, delay, cases_smooth, deaths_smooth, t):
        k = 10

        cases_past = cases_smooth[county, t - k - delay:t - delay]
        deaths_pred = self.models[c].predict(cases_past)
        deaths_curr = deaths_smooth[county, t - k:t]

        biased_values = []
        for i in range(deaths_curr.shape[0]):
            actual = deaths_curr[i]
            pred = deaths_pred[:, i]
            biased_quantile = self.interpolate_percentiles(actual, pred, True)
        biased_values.append(biased_quantile)
        m = np.mean(biased_values)
        s = np.std(biased_values)

        biased_quantiles = []
        for q in self.quantiles:
            q /= 10
            bq = norm.ppf(q, loc=m, scale=s)
            biased_quantiles.append(q)

        return np.array(biased_quantiles)

    def adjust_quantiles(self, pred_quantiles, biased_quantiles):
        adjusted_quantiles = []
        for i in range(pred_quantiles.shape[1]):
            pred_values = pred_quantiles[:, i]
            biased = self.interpolate_percentiles(biased_quantiles, 
                pred_values, False)

            adjusted = np.vstack([biased, pred_values])
            adjusted = np.mean(adjusted, axis=0)
            adjusted_quantiles.append(adjusted)

        return np.array(adjusted_quantiles)

    def predict(self, cases_smooth, deaths_smooth, horizon):
        self.predictions = {}
        t = len(self.cases_all) - self.val_steps
        for county in range(len(self.cases_all)):
            c = self.cluster_ids[county]
            delay = self.best_delays[c]
            cases_past = cases_smooth[county, t - delay:t - delay + horizon]
            deaths_pred_quantiles = self.models[c].predict(cases_past)

            biased_quantiles = self.get_bias(county, c, delay, cases_smooth,
                deaths_smooth, t)

            adjusted_pred = self.adjust_quantiles(deaths_pred_quantiles, 
                biased_quantiles)

            self.predictions[county] = adjusted_pred

        return self.predictions


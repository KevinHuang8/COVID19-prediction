import numpy as np
import scipy.stats as stats
from scipy.optimize import curve_fit

class CurvefitModel:
    def __init__(self, func, bounds):
        self.func = func
        self.bounds = bounds
        self.degenerate = False

    def fit(self, X_train, y_train):
        if not y_train.any():
            self.degenerate = True
            return
        try:
            self.popt, self.pcov = curve_fit(self.func, X_train, y_train, 
                bounds=self.bounds)
        except RuntimeError:
            self.degenerate = True
            return

    def predict(self, x, params=None):
        if self.degenerate:
            return np.zeros((x.shape[0],))
        if params is None:
            return self.func(x, *self.popt)
        else:
            return self.func(x, *params)

    def predict_quantiles(self, x, quantiles, samples=100, county=0):
        if self.degenerate:
            return np.array([np.zeros(x.shape[0]) 
                for i in range(len(quantiles))]).T
        
        errors = np.sqrt(np.diag(self.pcov))
        ## High errors for pre-peak/mid-peak counties
        ## Make errors max 1/2 of mean value
        count = 0
        while np.any(errors > (1/2) * self.popt):
            errors = errors / 2
            count += 1
            if count > 50:
                errors = np.zeros(self.popt.shape[0])
                break

        all_samples = []
        for i in range(samples):
            sample_params = np.random.normal(loc=self.popt, scale=errors)

            for i, param in enumerate(sample_params):
                lower_bound = self.bounds[0][i]
                upper_bound = self.bounds[1][i]

                epsilon = 1e-5
                if param < lower_bound:
                    sample_params[i] = lower_bound + epsilon
                elif param > upper_bound:
                    sample_params[i] = upper_bound

            y = self.predict(x, sample_params)
            if (np.any(np.isnan(y))):
                continue
            all_samples.append(y)

        all_samples = np.array(all_samples)
        quantile_predictions = np.array([np.percentile(all_samples, p, axis=0) 
            for p in quantiles])
        quantile_predictions = quantile_predictions.T
        quantile_predictions[quantile_predictions < 0] = 0
        return quantile_predictions

class ExpNormModel(CurvefitModel):
    def exp_model(x, max_val, loc, scale, K):
        return max_val*stats.exponnorm.pdf(x, K, loc, scale)

    def exp_model_cdf(x, max_val, loc, scale, K):
        return max_val*stats.exponnorm.cdf(x, K, loc, scale)

    def __init__(self, data_max):
        func = ExpNormModel.exp_model
        #max, loc, scale, K
        bounds = ([data_max, 0, 0, 0],
                       [100*data_max, np.inf, np.inf, 10])
        self.is_cumulative = False
        super().__init__(func, bounds)

    def set_cumulative(self, use_cumulative):
        if use_cumulative:
            self.is_cumulative = True
            self.func = ExpNormModel.exp_model_cdf
        else:
            self.is_cumulative = False
            self.func = ExpNormModel.exp_model

    def predict(self, x, params=None):
        p = super().predict(x, params)
        if self.is_cumulative:
            p = np.diff(p)
        return p
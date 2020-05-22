import numpy as np
import pymc3 as pm
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
        # High errors for pre-peak/mid-peak counties
        count = 0
        threshold = 2
        while np.any(errors > threshold * self.popt):
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

                if param < lower_bound:
                    sample_params[i] = lower_bound
                elif param > upper_bound:
                    sample_params[i] = upper_bound

            y = self.predict(x, sample_params)
            if (np.any(np.isnan(y))):
                continue
            all_samples.append(y)

        all_samples = np.array(all_samples)
        quantile_predictions = np.array([np.percentile(all_samples, p, axis=0) 
            for p in quantiles])
        ## Fudge arbitrarily
        quantile_predictions[0] *= 0.5
        quantile_predictions[1] *= 0.625
        quantile_predictions[2] *= 0.75
        quantile_predictions[3] *= 0.875
        ##
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

class GPModel(ExpNormModel):
    def fit(self, X_train, y_train, unsmoothed_y_train, **gp_params):
        super().fit(X_train, y_train)

        if self.degenerate:
            return

        y = self.predict(X_train)

        residuals = unsmoothed_y_train - y

        try:
            trials = gp_params['trials']
        except KeyError:
            trials = 5

        outputs = []
        for i in range(trials):
            output = self.fit_gp(X_train, residuals, **gp_params)
            outputs.append(output)
        outputs = np.array(outputs)

        self.quantiles = np.mean(outputs, axis=0)

    def fit_gp(self, X_train, residuals, **gp_params):
        s = np.std(residuals)
        m = np.mean(residuals)

        size = 100
        if self.is_cumulative:
            noise = np.random.normal(loc=m, scale=s, size=size - 1)
        else:
            noise = np.random.normal(loc=m, scale=s, size=size)
        X_train2 = np.linspace(X_train[0], X_train[-1] + 25, size)
        y_train2 = self.predict(X_train2) + noise
        if self.is_cumulative:
            X_train2 = X_train2[:-1]

        try:
            draw = gp_params['draw']
        except KeyError:
            draw = 500
        try:
            tune = gp_params['tune']
        except KeyError:
            tune = 500
        try:
            samples = gp_params['samples']
        except KeyError:
            samples = 100            

        with pm.Model() as gp_model:

            # Lengthscale
            ρ = pm.HalfCauchy('ρ', 5)
            η = pm.HalfCauchy('η', 5)
            
            M = ExponentialGaussianMean(*self.popt)
            K = (η**2) * pm.gp.cov.ExpQuad(1, ρ) 
            
            σ = pm.HalfNormal('σ', 50)
            
            expnorm_gp = pm.gp.Marginal(mean_func=M, cov_func=K)
            expnorm_gp.marginal_likelihood('expnorm', X=X_train2.reshape(-1,1), 
                                   y=y_train2, noise=σ)

        with gp_model:
            expnorm_gp_trace = pm.sample(draw, tune=tune, cores=1, 
                random_seed=42)

        self.X_pred = np.arange(0, np.max(X_train2) + 30)

        with gp_model:
            expnorm_deaths_pred = expnorm_gp.conditional('expnorm_deaths_pred2', 
                self.X_pred.reshape(-1, 1), pred_noise=True)
            gp_samples = pm.sample_posterior_predictive(expnorm_gp_trace, 
                vars=[expnorm_deaths_pred], samples=samples, random_seed=42)

        percentiles = [p for p in range(10, 100, 10)]
        quantile_gp = [np.percentile(gp_samples['expnorm_deaths_pred2'], q, 
            axis=0) for q in percentiles]
        quantile_gp = np.array(quantile_gp)

        quantile_gp[quantile_gp < 0] = 0

        return quantile_gp

    def predict(self, x, params=None):
        return super().predict(x, params)

    def predict_quantiles(self, x, quantiles, *args):
        if self.degenerate:
            return np.array([np.zeros(x.shape[0]) 
                for i in range(len(quantiles))]).T

        quantile_predictions = []
        for q in quantiles:
            y = self.quantiles[q][x]
            quantile_predictions.append(y)

        quantile_predictions = np.array(quantile_predictions)
        quantile_predictions = quantile_predictions.T
        quantile_predictions[quantile_predictions < 0] = 0
        return quantile_predictions

class ExponentialGaussianMean(pm.gp.mean.Mean):

    def __init__(self, max_val, loc, scale, K):
        pm.gp.mean.Mean.__init__(self)
        self.max_val = max_val
        self.loc = loc
        self.scale = scale
        self.K = K

    def __call__(self, X):
        return self.max_val*stats.exponnorm.pdf(X[0], self.K, self.loc, 
            self.scale)
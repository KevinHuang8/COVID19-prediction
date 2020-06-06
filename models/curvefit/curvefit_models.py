import numpy as np
import pymc3 as pm
import scipy.stats as stats
from scipy.optimize import curve_fit

class CurvefitModel:
    '''
    Base model class for least squares curve fitting.
    '''
    def __init__(self, func, bounds):
        '''
        func - the function to fit
        bounds - bounds for the parameters, in the format
        to pass into scipy.optimize.curve_fit
        '''
        self.func = func
        self.bounds = bounds
        # Degenerate flag signifies not enough data
        # Degenerate models should predict all zeros
        self.degenerate = False

    def fit(self, X_train, y_train):
        '''
        X_train, y_train - 1D np arrays

        Estimate parameters of the function given data.
        sef.popt will contain the parameter estimates that minimize
        mean squared error
        self.pcov will contain the estimated parameter covariances
        '''
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
        '''
        x - 1D np array

        Evaluates the function to x.

        If params is not None, then params are used as the parameters
        of the function, instead of the fitted parameters.
        '''
        if self.degenerate:
            return np.zeros((x.shape[0],))
        if params is None:
            return self.func(x, *self.popt)
        else:
            return self.func(x, *params)

    def predict_quantiles(self, x, quantiles, samples=100):
        '''
        x - 1D np array. Gives the locations to predict at
        quantiles - percentiles to compute
        samples - number of samples to take to estimate the quantiles

        Returns a 2D np array where axis 0 represents each x value
        and axis 1 represents each predicted quantile.

        We estimate quantiles by taking the error in the
        parameter estimates and creating a parameter distribution.
        We then sample the distribution, and take percentiles
        '''
        if self.degenerate:
            return np.array([np.zeros(x.shape[0]) 
                for i in range(len(quantiles))]).T
        
        errors = np.sqrt(np.diag(self.pcov))
        # High errors for pre-peak/mid-peak counties
        # Make sure the uncertainty does not exceed
        # threshold * actual value
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
        quantile_predictions = quantile_predictions.T
        # Remove spurrious predictions
        quantile_predictions[quantile_predictions < 0] = 0
        return quantile_predictions

class ExpNormModel(CurvefitModel):
    '''
    A model that uses an exponentially modified Gaussian curve.

    There are two modes: to use the exponentially modified Gaussian curve
    directly (meant for daily death data), or to use the cumulative
    exponentially modified Gaussian curve (meant for cumulative death data).

    In the latter case, results are always converted to the non-cumulative
    version.
    '''
    def exp_model(x, max_val, loc, scale, K):
        return max_val*stats.exponnorm.pdf(x, K, loc, scale)

    def exp_model_cdf(x, max_val, loc, scale, K):
        return max_val*stats.exponnorm.cdf(x, K, loc, scale)

    def __init__(self, data_max):
        func = ExpNormModel.exp_model
        # bounds: max, loc, scale, K
        # max value is between current max and 100 times the current max
        # K is empirically capped at 10 as that has been found to give good
        # results. This is a hyperparameter, but from empirical results needs
        # to be upper bounded by a small number (though not too small).
        bounds = ([data_max, 0, 0, 0],
                       [100*data_max, np.inf, np.inf, 10])
        # This flag is True when the cumulative curve should be used
        self.is_cumulative = False
        super().__init__(func, bounds)

    def set_cumulative(self, use_cumulative):
        '''
        use_cumulative - a boolean

        Changes the mode of the model into cumulative vs. non-cumulative mode.
        '''
        if use_cumulative:
            self.is_cumulative = True
            self.func = ExpNormModel.exp_model_cdf
        else:
            self.is_cumulative = False
            self.func = ExpNormModel.exp_model

    def predict(self, x, params=None):
        p = super().predict(x, params)
        # Cumulative results are always converted to the non-cumulative version
        # As a result, compared to the non-cumulative version, one extra data
        # point should be included at the end of x to align the two versions.
        if self.is_cumulative:
            p = np.diff(p)
        return p

class GPModel(ExpNormModel):
    '''
    This model uses a Gaussian process to refine the quantile estimates
    from the exponentially modified gaussian curve fit. It still uses the 
    same curve, but then simulates data points based on that curve fit and then
    trains a GP on those simulated data points.
    '''
    def fit(self, X_train, y_train, unsmoothed_y_train, **gp_params):
        '''
        unsmoothed_y_train - 1D array of the ground truth data points without
        any smoothing
        gp_params - to be passed into the GP, can contain (all optional)
            - horizon: how many steps to predict in the future
            - draw/tune: parameters for GP marginal likelihood sampling
            - samples: number of samples to take when estimating quantiles
            - trials: number of GP runs to average over
        '''
        super().fit(X_train, y_train)

        if self.degenerate:
            return

        y = self.predict(X_train)

        residuals = unsmoothed_y_train - y

        # We average results across 'trials' runs.
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
        '''
        We use the residuals to create a distribution of the noise of the
        exponentially modified Guassian prediction. We then hallucinate
        data points based on our curve fit prediction with added noise, then
        fit a GP.
        '''
        s = np.std(residuals)
        m = np.mean(residuals)

        try:
            horizon = gp_params['horizon']
        except KeyError:
            horizon = 30

        size = 100
        if self.is_cumulative:
            noise = np.random.normal(loc=m, scale=s, size=size - 1)
        else:
            noise = np.random.normal(loc=m, scale=s, size=size)
        X_train2 = np.linspace(X_train[0], X_train[-1] + horizon, size)
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
            expnorm_deaths_pred = expnorm_gp.conditional('expnorm_pred', 
                self.X_pred.reshape(-1, 1), pred_noise=True)
            gp_samples = pm.sample_posterior_predictive(expnorm_gp_trace, 
                vars=[expnorm_deaths_pred], samples=samples, random_seed=42)

        percentiles = [p for p in range(10, 100, 10)]
        quantile_gp = [np.percentile(gp_samples['expnorm_pred'], q, 
            axis=0) for q in percentiles]
        quantile_gp = np.array(quantile_gp)

        return quantile_gp

    def predict_quantiles(self, x, quantiles, *args):
        if self.degenerate:
            return np.array([np.zeros(x.shape[0]) 
                for i in range(len(quantiles))]).T

        quantile_predictions = []
        for q in range(len(quantiles)):
            y = self.quantiles[q][x]
            quantile_predictions.append(y)

        quantile_predictions = np.array(quantile_predictions)
        quantile_predictions = quantile_predictions.T
        quantile_predictions[quantile_predictions < 0] = 0
        return quantile_predictions

class ExponentialGaussianMean(pm.gp.mean.Mean):
    '''
    We use an exponentially modified Gaussian as the mean function of the GP.
    '''
    def __init__(self, max_val, loc, scale, K):
        pm.gp.mean.Mean.__init__(self)
        self.max_val = max_val
        self.loc = loc
        self.scale = scale
        self.K = K

    def __call__(self, X):
        return self.max_val*stats.exponnorm.pdf(X[0], self.K, self.loc, 
            self.scale)
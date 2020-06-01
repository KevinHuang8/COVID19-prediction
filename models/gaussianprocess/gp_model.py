import numpy as np
import pymc3 as pm
from ..utils import dataloader as loader
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import random

# import warnings
# warnings.simplefilter('ignore')

def linear(x, m):
    return x * m

def degenerate(x):
    return np.zeros(x.shape)

class GPCasesDeathsModel:
    def __init__(self, **params):
        try:
            self.draws = params['draws']
        except KeyError:
            self.draws = 100
        try:
            self.tune = params['tune']
        except KeyError:
            self.tune = 200
        try:
            self.samples = params['samples']
        except KeyError:
            self.samples = 100

    def scale_data(self, cases_past, deaths_curr):
        self.scale_factor = np.max(cases_past) / 100
        deaths_curr2 = np.array(deaths_curr) / self.scale_factor
        cases_past2 = np.array(cases_past) / self.scale_factor
        return cases_past2, deaths_curr2

    def fit(self, cases_past, deaths_curr, 
        quantiles=[10, 20, 30, 40, 50, 60, 70, 80, 90]):

        if len(cases_past) < 5:
            self.quantile_gp = []
            for q in quantiles:
                self.quantile_gp.append(degenerate)
            return self.quantile_gp

        cases_past2, deaths_curr2 = self.scale_data(cases_past, deaths_curr)

        mfit = curve_fit(linear, cases_past2, deaths_curr2)
        slope = mfit[0]

        with pm.Model() as gp_model:

            ρ = pm.HalfCauchy('ρ', 5)
            η = pm.HalfCauchy('η', 5)
            
            M = pm.gp.mean.Linear(coeffs=slope)
            K = (η**2) * pm.gp.cov.ExpQuad(1, ρ)
            
            σ = pm.HalfNormal('σ', 50)
                        
            deaths_gp = pm.gp.Marginal(mean_func=M, cov_func=K)
            deaths_gp.marginal_likelihood('deaths', X=cases_past2.reshape(-1,1),
                                   y=deaths_curr2, noise=σ)

        with gp_model:
            gp_trace = pm.sample(self.draws, tune=self.tune, cores=1,
                random_seed=random.randint(30, 80))

        X_pred = np.arange(0, np.max(cases_past2)*5)
        with gp_model:
            deaths_pred = deaths_gp.conditional("deaths_pred_noise", 
                X_pred.reshape(-1, 1), pred_noise=True)
            gp_samples = pm.sample_posterior_predictive(gp_trace, 
                vars=[deaths_pred], samples=self.samples)

        quantile_gp = [np.percentile(
            gp_samples['deaths_pred_noise'] * self.scale_factor, q, axis=0) 
                for q in quantiles]

        X_pred2 = X_pred * self.scale_factor
        self.quantile_gp = []
        for i in range(len(quantiles)):
            f = interp1d(X_pred2, quantile_gp[i], bounds_error=False,
                fill_value='extrapolate')
            self.quantile_gp.append(f)

    def predict(self, cases_past):
        deaths_curr = []
        for f in self.quantile_gp:
            deaths_curr.append(f(cases_past))
        return np.array(deaths_curr)

    def save(self, filename, folder):
        loader.save_to_otherdata(self.quantile_gp, filename, folder)

    def load(self, filename, folder):
        self.quantile_gp = loader.load_from_otherdata(filename, folder)
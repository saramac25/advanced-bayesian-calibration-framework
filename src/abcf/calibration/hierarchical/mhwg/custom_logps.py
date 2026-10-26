"""Log-densities of the blocks of the hierarchical sampler.

Original author: Gustav Robertson (Uppsala University) (created 27 Aug 2024), version used for
the time-dependent QoI.
Changes for abcf: demo block under __main__ removed (it used an older
signature of ``logp_X``); unused imports removed.

logp_hyper : sum_e sum_d log N(X[e, d] | mu_d, sig_d)   (hierarchical term)
logp_mu, logp_sig, logp_mu_sig(_d) : logp_hyper with uniform bounds
logp_X     : hierarchical term of case e + Gaussian likelihood
             N(meas | pred(x), meas_var + var_GP(x)); ``meas_var`` may be a
             vector (independent points) or a full covariance matrix.
"""

import numpy as np
from scipy.stats import multivariate_normal


def norm_logpdf(loc, scale, x):
    # Log-density of a normal (faster than scipy)
    return -np.log(scale) - .5 * np.log(2 * np.pi) - (loc - x) ** 2 / (2 * scale ** 2)


class logp_hyper:

    def __init__(self, lb=-np.inf, ub=np.inf):
        self.lb = lb
        self.ub = ub

    def __call__(self, state):
        return np.sum(norm_logpdf(state.mu, state.sig, state.X))

    def _check_bounds(self, param, lb, ub):
        return (param > ub).any() or (param < lb).any()


class logp_mu_sig(logp_hyper):

    def __call__(self, state):
        mu_sig = np.concatenate((state.mu, state.sig))
        if self._check_bounds(mu_sig, self.lb, self.ub):
            return -np.inf
        return super().__call__(state)


class logp_mu_sig_d(logp_hyper):

    def __init__(self, d, lb=-np.inf, ub=np.inf):
        self.d = d
        super().__init__(lb, ub)

    def __call__(self, state):
        mu_sig = np.array([state.mu[self.d], state.sig[self.d]])
        if self._check_bounds(mu_sig, self.lb, self.ub):
            return -np.inf
        return super().__call__(state)


class logp_mu(logp_hyper):

    def __call__(self, state):
        if self._check_bounds(state.mu, self.lb, self.ub):
            return -np.inf
        return super().__call__(state)


class logp_sig(logp_hyper):

    def __call__(self, state):
        if self._check_bounds(state.sig, self.lb, self.ub):
            return -np.inf
        return super().__call__(state)


class logp_X:

    def __init__(self, nexp, metamodel, meas, meas_var, lb=-np.inf, ub=np.inf, transform=None):
        # Case number, metamodel and optional transform of the parameters
        self.nexp = nexp
        self.metamodel = metamodel
        self.transform = transform
        # Measurement and measurement uncertainty
        self.meas = meas
        self.meas_var = meas_var
        self.lb = lb
        self.ub = ub

    def __call__(self, state):
        mu, sig, X = state.mu, state.sig, state.X
        x = X[self.nexp]

        if np.any(x < self.lb) or np.any(x > self.ub):
            return -np.inf

        # Hierarchical part (terms independent of x dropped)
        logp = np.sum(-(x - mu) ** 2 / (2 * sig ** 2))

        if self.transform is not None:
            x = self.transform(x)

        # Prediction and predictive variance of the metamodel
        pred, pred_var = self.metamodel.predict(x[None, :], return_cov=True)
        pred = pred.flatten()
        pred_var = pred_var.flatten()

        if self.meas_var.ndim == 2:
            cov = np.diag(pred_var) + self.meas_var
            logp += multivariate_normal(self.meas, cov).logpdf(pred)
        else:
            # Include both prediction and measurement uncertainty
            scale = (pred_var + self.meas_var) ** .5
            logp += np.sum(norm_logpdf(self.meas, scale, pred))
        return logp

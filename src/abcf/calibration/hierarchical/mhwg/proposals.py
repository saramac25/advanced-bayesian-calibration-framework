"""Gaussian random-walk proposals for the blocks of the hierarchical sampler.

Original author: Gustav Robertson (created 20 Sep 2024), version used for
the time-dependent QoI.
Changes for abcf:
  * ``propose_mu_d`` / ``propose_sig_d``: the 1-element proposal is
    assigned with ``[0]``. Assigning a length-1 array to a scalar slot is
    deprecated in NumPy 1.25 and an error in recent NumPy 2 versions;
    the result is identical.

The proposal covariance actually used is ``cov * adaptation_factor``
(initial adaptation_factor = 1.0): ``adapt_acceptance_rate`` tunes the
factor, ``adapt_covariance_matrix`` replaces ``cov``.
"""

import copy

import numpy as np


class gauss_proposal:

    def __init__(self, cov, **kwargs):
        self.cov = cov
        self.rng = kwargs.get('rng', np.random.default_rng())
        self.adaptation_factor = 1.0

    def set_adaptation_factor(self, adaptation_factor):
        self.adaptation_factor = adaptation_factor

    def _propose(self, value):
        """Generate a proposal for a given value (mu, sig, etc.)."""
        return self.rng.multivariate_normal(value, self.cov * self.adaptation_factor)

    def __call__(self, state):
        return self._propose(state)


class propose_mu(gauss_proposal):

    def __call__(self, state):
        candidate = copy.deepcopy(state)
        candidate.mu = self._propose(state.mu)
        return candidate


class propose_sig(gauss_proposal):

    def __call__(self, state):
        candidate = copy.deepcopy(state)
        candidate.sig = self._propose(state.sig)
        return candidate


class propose_mu_d(gauss_proposal):

    def __init__(self, cov, d, **kwargs):
        self.d = d
        super().__init__(cov, **kwargs)

    def __call__(self, state):
        candidate = copy.deepcopy(state)
        candidate.mu[self.d] = self._propose(np.atleast_1d(state.mu[self.d]))[0]
        return candidate


class propose_sig_d(gauss_proposal):

    def __init__(self, cov, d, **kwargs):
        self.d = d
        super().__init__(cov, **kwargs)

    def __call__(self, state):
        candidate = copy.deepcopy(state)
        candidate.sig[self.d] = self._propose(np.atleast_1d(state.sig[self.d]))[0]
        return candidate


class propose_mu_sig_d(gauss_proposal):

    def __init__(self, cov, d, **kwargs):
        self.d = d
        super().__init__(cov, **kwargs)

    def __call__(self, state):
        candidate = copy.deepcopy(state)
        mu_sig = np.array([state.mu[self.d], state.sig[self.d]])
        candidate.mu[self.d], candidate.sig[self.d] = self._propose(mu_sig)
        return candidate


class propose_hyper(gauss_proposal):

    def __call__(self, state):
        candidate = copy.deepcopy(state)
        hyper_candidate = self._propose(np.concatenate((state.mu, state.sig)))
        half = len(hyper_candidate) // 2
        candidate.mu = hyper_candidate[:half]
        candidate.sig = hyper_candidate[half:]
        return candidate


class propose_X(gauss_proposal):

    def __init__(self, cov, nexp, **kwargs):
        self.nexp = nexp
        super().__init__(cov, **kwargs)

    def __call__(self, state):
        candidate = copy.deepcopy(state)
        candidate.X[self.nexp] = self._propose(state.X[self.nexp])
        return candidate

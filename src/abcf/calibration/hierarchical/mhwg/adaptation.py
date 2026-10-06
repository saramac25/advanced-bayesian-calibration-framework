"""Adaptation of the proposal distributions during the burn-in.

Original author: Gustav Robertson (created 8 Oct 2024).
Changes for abcf:
  * class ``adapt_covariance_matrix_old`` (superseded) removed;
  * ``adapt_acceptance_rate``: new optional keywords ``adapt_start``,
    ``adapt_rate``, ``target_acceptance_rate`` and ``window``. Their
    defaults (1, 0.01, 0.44, None) reproduce the original behaviour
    exactly. ``window = n`` uses the acceptance rate of the last n steps
    instead of the cumulative one, which reacts faster: the cumulative
    rate keeps the memory of the first steps and makes the scale
    overshoot;
  * ``np.concat`` replaced by ``np.concatenate`` (works with NumPy < 2).

Covariance adaptation: adaptive Metropolis of Haario, Saksman &
Tamminen (2001), Bernoulli 7(2), 223-242, doi:10.2307/3318737, with the
running covariance updated by Welford's algorithm. The proposal
covariance is s_d * (C + epsilon I), default s_d = 2.4^2 / d.
Acceptance-rate adaptation: the scale of the proposal is multiplied by
exp(rate * (AR - target)) at every step; target 0.44 for 1D updates
(Roberts & Rosenthal, 2001, Statistical Science 16(4), 351-367,
doi:10.1214/ss/1015346320).
"""

import numpy as np


class adapt_covariance_matrix:

    def __init__(self, **kwargs):
        self.adapt_start = kwargs.get('adapt_start', 200)
        self.proposal_update_start = kwargs.get('proposal_update_start', 1000)
        self.adapt_stop = kwargs.get('adapt_stop', 3000)
        self.epsilon = kwargs.get('epsilon', 1e-6)
        self.s_d = kwargs.get('s_d', None)

        self.mean = None
        self.cov = None
        self.ndim = None
        self._n = 0  # counter for updates

    def __call__(self, sampler, proposal, state):
        sample_number = sampler._sample_number

        if sample_number < self.adapt_start or sample_number > self.adapt_stop:
            return

        state = np.asarray(state)

        if self.mean is None:
            self.ndim = len(state)
            self.mean = state.copy()
            self.cov = np.zeros((self.ndim, self.ndim))
            if self.s_d is None:
                self.s_d = 2.4 ** 2 / self.ndim
            self._n = 1
            return

        # Update mean and covariance (Welford's algorithm)
        self._n += 1
        delta = state - self.mean
        self.mean += delta / self._n
        self.cov += np.outer(delta, state - self.mean)

        # Update proposal covariance if allowed
        if sample_number >= self.proposal_update_start:
            cov_est = self.cov / (self._n - 1)
            proposal.cov = self.s_d * (cov_est + self.epsilon * np.eye(self.ndim))


class adapt_covariance_matrix_mu(adapt_covariance_matrix):

    def __call__(self, sampler, proposal, state):
        super().__call__(sampler, proposal, state.mu)


class adapt_covariance_matrix_sig(adapt_covariance_matrix):

    def __call__(self, sampler, proposal, state):
        super().__call__(sampler, proposal, state.sig)


class adapt_covariance_matrix_mu_d(adapt_covariance_matrix):

    def __init__(self, d, **kwargs):
        self.d = d
        super().__init__(**kwargs)

    def __call__(self, sampler, proposal, state):
        super().__call__(sampler, proposal, np.atleast_1d(state.mu[self.d]))


class adapt_covariance_matrix_sig_d(adapt_covariance_matrix):

    def __init__(self, d, **kwargs):
        self.d = d
        super().__init__(**kwargs)

    def __call__(self, sampler, proposal, state):
        super().__call__(sampler, proposal, np.atleast_1d(state.sig[self.d]))


class adapt_covariance_matrix_mu_sig_d(adapt_covariance_matrix):

    def __init__(self, d, **kwargs):
        self.d = d
        super().__init__(**kwargs)

    def __call__(self, sampler, proposal, state):
        mu_sig = np.array([state.mu[self.d], state.sig[self.d]])
        super().__call__(sampler, proposal, mu_sig)


class adapt_covariance_matrix_hyper(adapt_covariance_matrix):

    def __call__(self, sampler, proposal, state):
        state_hyper = np.concatenate((state.mu, state.sig))
        super().__call__(sampler, proposal, state_hyper)


class adapt_covariance_matrix_X(adapt_covariance_matrix):

    def __init__(self, nexp, **kwargs):
        self.nexp = nexp
        super().__init__(**kwargs)

    def __call__(self, sampler, proposal, state):
        super().__call__(sampler, proposal, state.X[self.nexp])


class adapt_acceptance_rate:

    def __init__(self, **kwargs):
        # Adaptation window [adapt_start, burnin], rate and target acceptance rate
        self.burnin = kwargs.get('burnin', 500)
        self.adapt_start = kwargs.get('adapt_start', 1)
        self.adapt_rate = kwargs.get('adapt_rate', 0.01)
        self.target_acceptance_rate = kwargs.get('target_acceptance_rate', 0.44)
        # None: cumulative acceptance rate (original); n: last n steps
        self.window = kwargs.get('window', None)
        self._recent = []

    def __call__(self, sampler, proposal, state):
        sample_number = sampler._sample_number

        if self.window:
            self._recent.append(sampler._last_accepted)
            if len(self._recent) > self.window:
                self._recent.pop(0)

        # Adapt only inside the adaptation window
        if self.adapt_start <= sample_number <= self.burnin:
            if self.window:
                current_acceptance_rate = np.mean(self._recent)
            else:
                current_acceptance_rate = sampler.get_acceptancerate()
            proposal.adaptation_factor *= np.exp(
                self.adapt_rate * (current_acceptance_rate - self.target_acceptance_rate))

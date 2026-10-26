"""Conjugate Gibbs updates of the population parameters (alternative to MH).

Original author: Gustav Robertson (Uppsala University).
Changes for abcf: demo block under __main__ removed.

mean_sampler: mu | X, sig ~ N(mean(X), sig^2 / n_cases)   (flat prior on mu)
sig_sampler:  sig^2 | X, mu ~ Inv-Gamma(n_cases/2, sum((X-mu)^2)/2)
"""

import copy

import numpy as np


class mean_sampler:

    def __init__(self, **kwargs):
        self.rng = kwargs.get('rng', np.random.default_rng())

    def step(self, state):
        loc = np.mean(state.X, axis=0)
        scale = state.sig / (len(state.X)) ** .5
        new_state = copy.deepcopy(state)
        new_state.mu = self.rng.normal(loc, scale)
        return new_state


class sig_sampler:

    def __init__(self, **kwargs):
        self.rng = kwargs.get('rng', np.random.default_rng())

    def step(self, state):
        X = state.X
        mu = state.mu
        new_state = copy.deepcopy(state)
        alpha = len(X) / 2
        beta = np.sum((X - mu) ** 2, axis=0) / 2
        # 1 / Gamma(alpha, scale=1/beta) is an inverse-gamma draw
        var = 1 / self.rng.gamma(alpha, 1 / beta)
        new_state.sig = var ** .5
        return new_state

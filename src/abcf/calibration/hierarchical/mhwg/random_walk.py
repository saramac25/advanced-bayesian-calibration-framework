"""Metropolis-Hastings block sampler.

Original author: Gustav Robertson (Uppsala University) (created 26 Aug 2024).
Changes for abcf: the demo code that followed the class (executed at
import time and requiring corner, arviz and matplotlib) was removed.

Only symmetric proposals are allowed (no Hastings correction).
"""

import numpy as np


class mh_sampler():

    def __init__(self, proposal, logp, recompute_logp=False, burnin=500, **kwargs):
        self.recompute_logp = recompute_logp
        self.proposal = proposal
        self.logp = logp
        self.rng = kwargs.get('rng', np.random.default_rng())
        self.adaptation = kwargs.get('adaptation', None)

        self._number_accepted = 0
        self._current_logp = None
        self._sample_number = 1
        self._last_accepted = False
        self._burnin = burnin

    def step(self, state):
        current_logp = self._current_logp
        if current_logp is None or self.recompute_logp:
            current_logp = self.logp(state)

        new_state = self.proposal(state)
        new_logp = self.logp(new_state)

        ulog = np.log(self.rng.random())
        self._last_accepted = False

        # Non-symmetric moves are not allowed
        if new_logp - current_logp > ulog:
            state = new_state
            current_logp = new_logp
            self._number_accepted += 1
            self._last_accepted = True

        # If the proposal allows for adapting, adapt
        if self.adaptation is not None:
            self.adaptation(self, self.proposal, state)

        self._current_logp = current_logp
        self._sample_number += 1
        return state

    def get_acceptancerate(self):
        return self._number_accepted / self._sample_number

    def run_mcmc(self, state, nsteps=10000):
        for _ in range(nsteps):
            state = self.step(state)
            yield state

"""Posterior samples in physical units, summaries and predictive draws."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import truncnorm


class Posterior:
    """Post-burn-in posterior of one results file, in physical units."""

    def __init__(self, results: dict):
        md, ch, sc = results["metadata"], results["chains"], results["scaler"]
        self.method = md["method"]
        self.qoi_type = md["qoi_type"]
        self.names = list(md["param_names"])
        self.case_ids = list(md["case_ids"])
        self.idx = np.asarray(md["calibrated_idx"], dtype=int)
        self.scaler = sc
        self.burnin = int(md["burnin"])
        m, s = sc["mean"][self.idx], sc["std"][self.idx]
        self.lower_phys = sc["lower"][self.idx] * s + m
        self.upper_phys = sc["upper"][self.idx] * s + m
        b = self.burnin
        if self.method == "SC":
            self.theta_std = ch["theta"][:, b:, :].reshape(-1, len(self.names))
            self.theta = self.theta_std * s + m
        else:
            p = len(self.names)
            self.mu_std = ch["mu"][:, b:, :].reshape(-1, p)
            self.sigma_std = ch["sigma"][:, b:, :].reshape(-1, p)
            self.X_std = ch["X"][:, b:, :, :].reshape(-1, len(self.case_ids), p)
            self.mu = self.mu_std * s + m
            self.sigma = self.sigma_std * s
            self.X = self.X_std * s + m

    def population(self, n: int, seed: int = 0) -> np.ndarray:
        """HC: draws of a new case's parameters, theta ~ N(mu, sigma^2) truncated
        to the training range, with (mu, sigma) from the posterior (physical units)."""
        rng = np.random.default_rng(seed)
        k = rng.integers(0, len(self.mu), size=n)
        mu, sig = self.mu[k], self.sigma[k]
        a = (self.lower_phys - mu) / sig
        b = (self.upper_phys - mu) / sig
        return truncnorm(a, b, loc=mu, scale=sig).rvs(random_state=rng)

    def draws(self, n: int, seed: int = 0) -> np.ndarray:
        """Posterior draws of the calibrated parameters for a new case (physical units)."""
        if self.method == "SC":
            rng = np.random.default_rng(seed)
            return self.theta[rng.integers(0, len(self.theta), size=n)]
        return self.population(n, seed)

    def case_draws(self, case_index: int, n: int, seed: int = 0) -> np.ndarray:
        """Draws for a calibration case: theta for SC, X[case] for HC (physical units)."""
        rng = np.random.default_rng(seed)
        if self.method == "SC":
            return self.theta[rng.integers(0, len(self.theta), size=n)]
        return self.X[rng.integers(0, len(self.X), size=n), case_index]

    def summary(self) -> pd.DataFrame:
        def stats(block, arr):
            return [{"method": self.method, "block": block, "parameter": n,
                     "mean": arr[:, j].mean(), "std": arr[:, j].std(),
                     "q2.5": np.percentile(arr[:, j], 2.5), "median": np.median(arr[:, j]),
                     "q97.5": np.percentile(arr[:, j], 97.5)} for j, n in enumerate(self.names)]
        if self.method == "SC":
            return pd.DataFrame(stats("theta", self.theta))
        rows = stats("mu", self.mu) + stats("sigma", self.sigma) + stats("population", self.population(20000))
        for e, cid in enumerate(self.case_ids):
            rows += [dict(r, block=f"X[{cid}]") for r in stats("X", self.X[:, e, :])]
        return pd.DataFrame(rows)

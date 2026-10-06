"""Convergence diagnostics on post-burn-in chains.

* R-hat (rank-normalised, split) and bulk/tail ESS: ArviZ, following
  Vehtari et al. (2021), Bayesian Analysis 16(2), 667-718,
  doi:10.1214/20-BA1221.
* MCSE: overlapping batch means (OBM), Flegal & Jones (2010), Annals of
  Statistics 38(2), 1034-1070, doi:10.1214/09-AOS735. Implementation
  from the thesis scripts.
"""

from __future__ import annotations

import arviz as az
import numpy as np


def _rhat(x: np.ndarray) -> float:
    return float(az.rhat(np.asarray(x, dtype=float), method="rank"))


def _ess(x: np.ndarray, method: str) -> float:
    x = np.asarray(x, dtype=float)
    try:
        return float(az.ess(x, method=method))
    except TypeError:          # ArviZ >= 1.0 requires the tail quantiles explicitly
        return float(az.ess(x, method=method, prob=(0.05, 0.95)))


def mcse_obm_single(chain: np.ndarray, batch_size: int | None = None):
    """MCSE of the mean of a 1D chain by overlapping batch means.

    Returns (mcse, sigma2), sigma2 being the asymptotic variance estimate.
    Default batch size: floor(N^(1/3)), at least 2.
    """
    chain = np.asarray(chain, dtype=float)
    n = len(chain)
    b = max(int(np.floor(n ** (1 / 3))) if batch_size is None else int(batch_size), 2)
    n_batches = n - b + 1
    x_bar = chain.mean()
    cs = np.concatenate([[0.0], np.cumsum(chain)])
    batch_means = (cs[b:] - cs[:n_batches]) / b
    sigma2 = (n * b) / (n_batches * (n_batches - 1)) * np.sum((batch_means - x_bar) ** 2)
    return np.sqrt(sigma2 / n), sigma2


def mcse_obm(chains: np.ndarray, batch_size: int | None = None, pooled: bool = False):
    """MCSE for chains of shape (n_chains, n_draws).

    sigma2 is averaged over the chains. ``pooled=False`` divides by the
    draws of ONE chain (conservative, the choice of the thesis scripts:
    MCSE larger by sqrt(n_chains)); ``pooled=True`` divides by all draws.
    """
    chains = np.atleast_2d(np.asarray(chains, dtype=float))
    n_chains, n_draws = chains.shape
    sigma2 = np.mean([mcse_obm_single(c, batch_size)[1] for c in chains])
    n = n_draws * n_chains if pooled else n_draws
    return float(np.sqrt(sigma2 / n))


def chain_diagnostics(chains: np.ndarray, mcse_pooled: bool = False) -> dict:
    """Diagnostics per parameter for chains of shape (n_chains, n_draws, n_params)."""
    chains = np.asarray(chains, dtype=float)
    if chains.ndim == 2:
        chains = chains[:, :, None]
    n_params = chains.shape[2]
    out = {k: np.empty(n_params) for k in
           ("rhat", "ess_bulk", "ess_tail", "mean", "std", "mcse", "mcse_over_std")}
    for p in range(n_params):
        x = chains[:, :, p]
        out["rhat"][p] = _rhat(x)
        out["ess_bulk"][p] = _ess(x, "bulk")
        out["ess_tail"][p] = _ess(x, "tail")
        out["mean"][p] = x.mean()
        out["std"][p] = x.std()
        out["mcse"][p] = mcse_obm(x, pooled=mcse_pooled)
        out["mcse_over_std"][p] = out["mcse"][p] / out["std"][p] if out["std"][p] > 0 else np.nan
    return out


def acceptance_from_chain(x: np.ndarray) -> float:
    """Acceptance rate of a Metropolis chain: fraction of moves that changed the state."""
    x = np.asarray(x)
    if x.ndim == 1:
        return float(np.mean(np.diff(x) != 0))
    return float(np.mean(np.any(np.diff(x, axis=0) != 0, axis=-1)))

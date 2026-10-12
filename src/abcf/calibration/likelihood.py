"""Gaussian likelihood shared by the standard and hierarchical calibration.

For one case, with standardised calibrated parameters x:

    z_meas ~ N( z_GP(x),  W Sigma_meas W^T  [+ diag(var_GP(x))] )

* z = PCA scores of the QoI for a time-dependent QoI, the QoI itself for
  a scalar QoI (W = 1);
* Sigma_meas = diag( (relative_noise * |meas|)^2 ), relative measurement
  uncertainty set in the configuration;
* var_GP is the GP predictive variance, included when
  ``likelihood.include_gp_variance`` is true.
"""

from __future__ import annotations

import numpy as np

from ..config import ParameterSetup
from ..surrogate.gp import CaseSurrogate, InputScaler

LOG2PI = np.log(2.0 * np.pi)


class ParameterExpander:
    """Standardised calibrated vector -> standardised full vector.

    Non-calibrated parameters are fixed at their (standardised) fixed
    value. Generalises ``metamodel_wrapper._expand`` of the time-dependent
    calibration script.
    """

    def __init__(self, setup: ParameterSetup, scaler: InputScaler, case_id=None):
        self.idx = np.asarray(setup.calibrated_idx)
        full_phys = setup.full_vector(np.zeros(setup.n_calibrated), case_id)[0]
        self.x_fixed = scaler.transform(full_phys)

    def __call__(self, x_cal: np.ndarray) -> np.ndarray:
        x_cal = np.atleast_2d(x_cal)
        full = np.tile(self.x_fixed, (x_cal.shape[0], 1))
        full[:, self.idx] = x_cal
        return full


class CaseLikelihood:
    """Log-likelihood of one case as a function of the calibrated parameters."""

    def __init__(self, surrogate: CaseSurrogate, meas: np.ndarray, relative_noise: float,
                 expander: ParameterExpander, include_gp_variance: bool = True):
        self.surrogate = surrogate
        self.expander = expander
        self.include_gp_variance = include_gp_variance
        red = surrogate.reduction
        self.z_meas = red.project(meas)[0]                     # (k,)
        sigma_meas = relative_noise * np.abs(np.asarray(meas, dtype=float))
        self.meas_cov = red.project_covariance(np.diag(sigma_meas ** 2))   # (k, k)
        self.k = self.z_meas.shape[0]
        if not include_gp_variance:
            self._chol = np.linalg.cholesky(self.meas_cov)
            self._logdet = 2.0 * np.sum(np.log(np.diag(self._chol)))
        # Small cache of the last GP evaluations at single points: in a Gibbs sweep
        # the same x of a case is evaluated several times (current state of the X
        # block, joint shift move, stored log-posterior). Exact, not an approximation.
        self._cache = {}
        self._cache_size = 8

    def _gp(self, x_cal):
        """GP mean and variance (n, k) at calibrated x, cached for single points."""
        x_cal = np.atleast_2d(np.asarray(x_cal, dtype=float))
        if x_cal.shape[0] != 1:
            return self.surrogate.predict(self.expander(x_cal), return_var=True)
        key = x_cal.tobytes()
        hit = self._cache.get(key)
        if hit is None:
            hit = self.surrogate.predict(self.expander(x_cal), return_var=True)
            if len(self._cache) >= self._cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = hit
        return hit

    # ---- interface used by Gustav Robertson's ``logp_X`` ---------------- #
    def predict(self, x_cal, return_cov=True):
        """Reduced-space mean and variance, (n, k) each, from calibrated x."""
        mean, var = self._gp(x_cal)
        if not self.include_gp_variance:
            var = np.zeros_like(var)
        return (mean, var) if return_cov else mean

    # ---- direct evaluation --------------------------------------------- #
    def __call__(self, x_cal: np.ndarray) -> float:
        x_cal = np.atleast_2d(x_cal)
        mean, var = self._gp(x_cal)
        resid = self.z_meas - mean[0]
        if self.include_gp_variance:
            cov = self.meas_cov + np.diag(var[0])
            chol = np.linalg.cholesky(cov)
            logdet = 2.0 * np.sum(np.log(np.diag(chol)))
        else:
            chol, logdet = self._chol, self._logdet
        sol = np.linalg.solve(chol, resid)
        return float(-0.5 * (sol @ sol + logdet + self.k * LOG2PI))


def build_likelihoods(cfg: dict, data, sset, setup: ParameterSetup, case_ids: list) -> list:
    """One CaseLikelihood per calibration case, in the order of ``case_ids``."""
    from ..config import case_noise
    inc = bool(cfg["likelihood"]["include_gp_variance"])
    # one expander per case: the non-calibrated parameters may differ between cases
    return [CaseLikelihood(sset[cid], data.cases[cid].meas, case_noise(cfg, cid),
                           ParameterExpander(setup, sset.scaler, cid), inc)
            for cid in case_ids]

"""Predictive validation metrics.

For every case: the solver was re-run with N posterior samples, giving
predictions pred (N, n_t); meas (n_t,) is the measurement, with relative
uncertainty r (sigma_meas = r |meas|).

* z-score at each point: (meas - mean(pred)) / sqrt(var(pred) + sigma_meas^2)
* coverage: fraction of measured points inside the central predictive
  interval (level from the config), the predictive distribution
  including the measurement noise;
* RMSE between the predictive mean and the measurement.

Chi-square metrics (one value over the validation points, ideal value 1):

* chi2_E   = (1/N) sum (p_fit - y)^2 / sigma_exp^2
* chi2_tot = (1/N) sum (p_fit - y)^2 / (sigma_fit^2 + sigma_exp^2),
  sigma_fit^2 = sigma_post^2 + sigma_GP^2 (Helgesson et al., 2018;
  Robertson et al., 2026),

with p_fit the prediction at the MAP point, sigma_post the standard
deviation of the posterior predictions, sigma_GP the GP standard
deviation at the MAP point and sigma_exp the measurement uncertainty.
chi2_tot <= chi2_E always; sqrt(chi2_E) is the RMS deviation in units of
sigma_exp. Averaging (C_s - y)^2 / sigma_exp^2 over every posterior sample
is avoided on purpose: it penalises the width of the predictive band.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def case_metrics(pred: np.ndarray, meas: np.ndarray, relative_noise: float, level: float,
                 seed: int = 0) -> dict:
    pred = np.atleast_2d(pred)
    meas = np.asarray(meas, dtype=float)
    sigma_meas = relative_noise * np.abs(meas)
    mean = pred.mean(axis=0)
    var = pred.var(axis=0, ddof=1) if pred.shape[0] > 1 else np.zeros_like(mean)
    z = (meas - mean) / np.sqrt(var + sigma_meas ** 2)

    # Predictive samples with measurement noise, for the interval
    rng = np.random.default_rng(seed)
    noisy = pred + rng.standard_normal(pred.shape) * sigma_meas
    q_lo, q_hi = np.quantile(noisy, [(1 - level) / 2, (1 + level) / 2], axis=0)
    inside = (meas >= q_lo) & (meas <= q_hi)
    return {"z": z, "mean": mean, "q_lo": q_lo, "q_hi": q_hi, "inside": inside,
            "coverage": float(inside.mean()), "rmse": float(np.sqrt(np.mean((mean - meas) ** 2))),
            "mean_abs_z": float(np.mean(np.abs(z))), "max_abs_z": float(np.max(np.abs(z)))}


def chi2_terms(y, p_fit, sigma_post, sigma_gp, sigma_exp) -> dict:
    """Per-point terms of chi2_E and chi2_tot (arrays of equal length)."""
    y, p_fit = np.asarray(y, float), np.asarray(p_fit, float)
    sigma_fit2 = np.asarray(sigma_post, float) ** 2 + np.asarray(sigma_gp, float) ** 2
    se2 = np.asarray(sigma_exp, float) ** 2
    return {"chi2_E_term": (p_fit - y) ** 2 / se2,
            "chi2_tot_term": (p_fit - y) ** 2 / (sigma_fit2 + se2),
            "sigma_fit": np.sqrt(sigma_fit2)}


def validation_table(results: dict) -> pd.DataFrame:
    """results: method -> case_id -> metrics dict."""
    rows = []
    for method, cases in results.items():
        for cid, m in cases.items():
            rows.append({"method": method, "case": cid, "n_points": len(m["z"]),
                         "coverage": m["coverage"], "rmse": m["rmse"],
                         "mean_abs_z": m["mean_abs_z"], "max_abs_z": m["max_abs_z"]})
    return pd.DataFrame(rows)

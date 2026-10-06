"""Predictive validation metrics.

For every case: the solver was re-run with N posterior samples, giving
predictions pred (N, n_t); meas (n_t,) is the measurement, with relative
uncertainty r (sigma_meas = r |meas|).

* z-score at each point: (meas - mean(pred)) / sqrt(var(pred) + sigma_meas^2)
* coverage: fraction of measured points inside the central predictive
  interval (level from the config), the predictive distribution
  including the measurement noise;
* RMSE between the predictive mean and the measurement.
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


def validation_table(results: dict) -> pd.DataFrame:
    """results: method -> case_id -> metrics dict."""
    rows = []
    for method, cases in results.items():
        for cid, m in cases.items():
            rows.append({"method": method, "case": cid, "n_points": len(m["z"]),
                         "coverage": m["coverage"], "rmse": m["rmse"],
                         "mean_abs_z": m["mean_abs_z"], "max_abs_z": m["max_abs_z"]})
    return pd.DataFrame(rows)

"""Health checks of the hierarchical sampler (from the thesis script
``troubleshooting_test1.py``), generalised:

* acceptance rate of every X block after the burn-in, per chain;
* collapse of the proposal adaptation: the acceptance rate measured in
  successive windows drops towards zero while it was healthy earlier.
  For a collapsed case, a "healthy" proposal covariance is estimated
  from the window before the collapse, ready to be pasted into
  ``hierarchical.fixed_proposal_cov`` of the configuration;
* NaN values in the chains.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ar(seg: np.ndarray) -> float:
    """Acceptance rate of a (n_steps, p) segment of an X chain."""
    if len(seg) < 2:
        return np.nan
    return float(np.mean(np.any(np.diff(seg, axis=0) != 0, axis=1)))


def acceptance_table(X: np.ndarray, burnin: int, case_ids: list, param_names: list,
                     ar_range: tuple) -> pd.DataFrame:
    """X: (n_chains, n_steps, n_cases, p). One row per (chain, case)."""
    rows = []
    for c in range(X.shape[0]):
        for e, cid in enumerate(case_ids):
            seg = X[c, burnin:, e, :]
            ar = _ar(seg)
            rows.append({"chain": c + 1, "case": cid, "acceptance": ar,
                         "ok": bool(ar_range[0] <= ar <= ar_range[1]),
                         **{f"var_{n}": float(np.var(seg[:, j])) for j, n in enumerate(param_names)}})
    return pd.DataFrame(rows)


def collapse_check(X: np.ndarray, nsteps: int, case_ids: list, n_windows: int = 8,
                   healthy_ar: float = 0.15, collapse_ar: float = 0.05):
    """Look for adaptation collapse in every chain and case.

    Returns (table, suggestions) where suggestions maps case_id -> (p, p)
    covariance estimated in the healthy window that preceded the collapse
    (standardised units, as expected by fixed_proposal_cov).
    """
    edges = np.linspace(0, nsteps, n_windows + 1).astype(int)
    rows, suggestions = [], {}
    for c in range(X.shape[0]):
        for e, cid in enumerate(case_ids):
            ars = [_ar(X[c, a:b, e, :]) for a, b in zip(edges[:-1], edges[1:])]
            earlier_best = int(np.nanargmax(ars[:-1]))
            collapsed = ars[earlier_best] > healthy_ar and ars[-1] < collapse_ar
            row = {"chain": c + 1, "case": cid, "collapsed": bool(collapsed)}
            row.update({f"AR[{a}:{b}]": ar for a, b, ar in zip(edges[:-1], edges[1:], ars)})
            rows.append(row)
            if collapsed and cid not in suggestions:
                a, b = edges[earlier_best], edges[earlier_best + 1]
                seg = X[c, a:b, e, :]
                cov = np.atleast_2d(np.cov(seg, rowvar=False))
                # The proposal used by the sampler is cov * adaptation_factor (0.1 by default);
                # the suggestion is the empirical covariance scaled by 2.38^2/p (Haario).
                suggestions[cid] = cov * 2.38 ** 2 / seg.shape[1]
    return pd.DataFrame(rows), suggestions


def nan_check(X: np.ndarray, case_ids: list) -> list:
    """List of (chain, case) with NaN values."""
    return [(c + 1, cid) for c in range(X.shape[0]) for e, cid in enumerate(case_ids)
            if np.isnan(X[c, :, e, :]).any()]


def format_fixed_cov_yaml(suggestions: dict) -> str:
    """YAML snippet for hierarchical.fixed_proposal_cov."""
    if not suggestions:
        return ""
    lines = ["hierarchical:", "  fixed_proposal_cov:"]
    for cid, cov in suggestions.items():
        lines.append(f"    {cid}:")
        for row in np.atleast_2d(cov):
            lines.append("      - [" + ", ".join(f"{v:.4e}" for v in row) + "]")
    return "\n".join(lines)

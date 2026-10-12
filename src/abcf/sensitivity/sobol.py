"""Variance-based (Sobol') sensitivity indices computed on the GP surrogates.

For a QoI Y = f(X_1, ..., X_d) with uncertain inputs:

* first-order index  S1_i = Var(E[Y | X_i]) / Var(Y): fraction of the variance of Y due to
  X_i alone (expected reduction of the variance if X_i were known exactly);
* total index        ST_i = E[Var(Y | X_~i)] / Var(Y): fraction due to X_i alone plus all
  its interactions (variance left if every input except X_i were known). ST_i ~ 0 means
  that X_i can be fixed at any value of its range without changing the variance of Y.

Estimated with the Saltelli scheme (matrices A, B and A_B^(i), N (d + 2) evaluations) and
the Jansen estimator of ST (Saltelli et al. 2010, Comput. Phys. Commun. 181(2) 259-270,
doi:10.1016/j.cpc.2009.09.018; Sobol' 2001, Math. Comput. Simul. 55 271-280,
doi:10.1016/S0378-4754(00)00270-6), through SALib (Herman & Usher 2017, JOSS 2(9) 97,
doi:10.21105/joss.00097). The model is the GP mean of each case (Marrel et al. 2009,
Reliab. Eng. Syst. Saf. 94(3) 742-751, doi:10.1016/j.ress.2008.07.008), so the cost is a
few thousand GP predictions per case. For a time-dependent QoI the indices are computed
on the reconstructed output at every time and on its time average.

Input distributions (physical units), one entry per parameter of the surrogate set:
    {"type": "normal", "mean": m, "std": s}          truncated to the GP training range
    {"type": "uniform", "low": a, "high": b}
    {"type": "fixed", "value": v}                    not varied (excluded from the indices)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _sample(dists: dict, names: list, lower: np.ndarray, upper: np.ndarray, n: int, seed: int):
    from SALib.sample import sobol as sobol_sample
    var = [p for p in names if dists[p]["type"] != "fixed"]
    problem = {"num_vars": len(var), "names": var, "bounds": [[0.0, 1.0]] * len(var)}
    U = sobol_sample.sample(problem, n, calc_second_order=False, seed=seed)    # uniform [0, 1]
    U = np.clip(U, 1e-12, 1 - 1e-12)
    X = np.empty((U.shape[0], len(names)))
    from scipy import stats
    for j, p in enumerate(names):
        d = dists[p]
        lo, hi = lower[j], upper[j]
        if d["type"] == "fixed":
            X[:, j] = float(d["value"])
            continue
        u = U[:, var.index(p)]
        if d["type"] == "uniform":
            X[:, j] = float(d["low"]) + u * (float(d["high"]) - float(d["low"]))
        elif d["type"] == "normal":
            m, s = float(d["mean"]), float(d["std"])
            X[:, j] = stats.truncnorm.ppf(u, (lo - m) / s, (hi - m) / s, loc=m, scale=s)
        else:
            raise ValueError(f"{p}: unknown distribution type {d['type']!r}")
    return problem, X


def sobol_indices(sset, dists: dict, n_base: int = 1024, seed: int = 1, cases: list | None = None,
                  n_bootstrap: int = 200) -> pd.DataFrame:
    """S1 and ST of every varied parameter for every case (GP mean as the model).

    Returns one row per (case, parameter): S1, S1_conf, ST, ST_conf (95% bootstrap
    half-widths of SALib), var_Y (variance of the QoI, time-averaged for a time series).
    """
    from SALib.analyze import sobol as sobol_analyze
    names = list(sset.param_names)
    missing = [p for p in names if p not in dists]
    if missing:
        raise ValueError(f"No distribution given for {missing}")
    lower = sset.scaler.inverse(sset.scaler.lower)      # GP training range, physical units
    upper = sset.scaler.inverse(sset.scaler.upper)
    problem, X = _sample(dists, names, lower, upper, n_base, seed)
    Xs = sset.scaler.transform(X)
    rows = []
    for cid in (cases or list(sset.surrogates)):
        sur = sset[cid]
        y = np.asarray(sur.predict_qoi(Xs)).reshape(X.shape[0], -1)
        y_eval = y.mean(axis=1)                          # scalar QoI, or time average
        res = sobol_analyze.analyze(problem, y_eval, calc_second_order=False,
                                    num_resamples=n_bootstrap, seed=seed)
        for i, p in enumerate(problem["names"]):
            rows.append({"case": cid, "parameter": p, "S1": res["S1"][i], "S1_conf": res["S1_conf"][i],
                         "ST": res["ST"][i], "ST_conf": res["ST_conf"][i], "var_Y": float(np.var(y_eval))})
    return pd.DataFrame(rows)


def sobol_summary(table: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    """Across cases: mean and 2.5-97.5 percentiles of S1 and ST; suggestion "calibrate" if the
    mean ST >= threshold (the parameter explains at least that fraction of the variance),
    "fix" otherwise."""
    g = table.groupby("parameter")
    out = pd.DataFrame({
        "S1_mean": g["S1"].mean(), "S1_q2_5": g["S1"].quantile(0.025), "S1_q97_5": g["S1"].quantile(0.975),
        "ST_mean": g["ST"].mean(), "ST_q2_5": g["ST"].quantile(0.025), "ST_q97_5": g["ST"].quantile(0.975),
        "n_cases": g["case"].nunique()})
    out["interaction_ST_minus_S1"] = out["ST_mean"] - out["S1_mean"]
    out["suggestion"] = np.where(out["ST_mean"] >= threshold, "calibrate", "fix")
    return out.sort_values("ST_mean", ascending=False)

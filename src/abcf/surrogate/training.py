"""Training, testing and storage of the surrogates of all cases."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..io import InputData
from .gp import CaseSurrogate, InputScaler, SurrogateSet


def train_surrogates(data: InputData, cfg: dict, verbose: bool = True):
    """Fit one CaseSurrogate per case. Returns (SurrogateSet, summary DataFrame).

    The train/test split is done on the solver runs (eval_ids) with the
    project seed, so it is reproducible.
    """
    cfg_s = cfg["surrogate"]
    seed = int(cfg["project"]["seed"])
    scaler = InputScaler(data.inputs.to_numpy(dtype=float))

    surrogates, rows = {}, []
    for i, (cid, case) in enumerate(data.cases.items()):
        X = data.X(case)
        idx_train, idx_test = train_test_split(np.arange(X.shape[0]),
                                               test_size=cfg_s["test_fraction"],
                                               random_state=seed)
        Xs = scaler.transform(X)
        sur = CaseSurrogate(cid).fit(Xs[idx_train], case.pred[idx_train], cfg_s, seed=seed + 100 * i)
        metrics = sur.evaluate(Xs[idx_test], case.pred[idx_test])
        surrogates[cid] = sur
        row = {"case": cid, "n_t": case.n_t, "n_train": len(idx_train), "n_test": len(idx_test),
               "n_components": sur.n_components,
               "explained_variance": float(np.sum(sur.reduction.explained_variance_ratio)),
               "r2_test": metrics["r2"], "rmse_test": metrics["rmse"],
               "gp_coverage_95": metrics["coverage_95"],
               "alphas": ";".join(f"{a:g}" for a in sur.alphas),
               "hit_bound": any(sur.hit_bound),
               "kernels": " | ".join(str(gp.kernel_) for gp in sur.gps)}
        rows.append(row)
        if verbose:
            print(f"  {cid}: {sur.n_components} GP(s), R2_test = {metrics['r2']:.4f}, "
                  f"RMSE_test = {metrics['rmse']:.4g}, GP 95% coverage = {metrics['coverage_95']:.2f}")
    return SurrogateSet(scaler, surrogates, data.param_names), pd.DataFrame(rows)


def _split(case, X, cfg_s, seed):
    return train_test_split(np.arange(X.shape[0]), test_size=cfg_s["test_fraction"], random_state=seed)


def compare_kernels(data: InputData, cfg: dict, verbose: bool = True):
    """Train every kernel of ``surrogate.kernel_candidates`` on the same train/test
    split and score it on the test runs.

    Metrics per case and kernel (test runs):
      * rel_rmse_pct: RMSE of the QoI relative to the simulated value, in % (to be
        compared with the measurement uncertainty);
      * q2: predictivity coefficient 1 - SS_res / SS_tot;
      * std_res_mean / std_res_std: mean and std of the standardised residuals
        (y - m) / s_GP, ideally 0 and 1 (calibrated GP variance);
      * coverage_95: fraction of |standardised residual| <= 1.96, ideally 0.95;
      * nlpd: mean negative log predictive density, a proper scoring rule that
        rewards accuracy and well-sized variances together (Gneiting & Raftery,
        2007, doi:10.1198/016214506000001437); lower is better.
    The residuals and the NLPD are computed in the reduced (PCA) space.
    Returns (per-case table, summary table, dict kernel -> test arrays for plots).
    """
    import copy
    cfg_s = cfg["surrogate"]
    cands = cfg_s.get("kernel_candidates") or {}
    if not cands:
        raise ValueError("surrogate.kernel_candidates is empty: nothing to compare.")
    seed = int(cfg["project"]["seed"])
    scaler = InputScaler(data.inputs.to_numpy(dtype=float))
    sel = cfg_s.get("kernel_comparison_cases", "all")
    case_ids = data.case_ids if sel == "all" else list(sel)
    rows, arrays = [], {}
    for name, kspec in cands.items():
        c = copy.deepcopy(cfg_s)
        for key in ("alpha_grid", "alpha_fixed", "n_restarts_optimizer"):
            if key in kspec:
                c[key] = kspec[key]
        c["kernel"] = {**cfg_s["kernel"], **{k: v for k, v in kspec.items()
                                            if k in ("terms", "combine", "matern_nu", "white_noise",
                                                     "length_scale_bounds")}}
        arr = {"y": [], "m": [], "z": []}
        for i, cid in enumerate(case_ids):
            case = data.cases[cid]
            X = data.X(case)
            tr, te = _split(case, X, c, seed)
            Xs = scaler.transform(X)
            sur = CaseSurrogate(cid).fit(Xs[tr], case.pred[tr], c, seed=seed + 100 * i)
            y = case.pred[te]
            m = sur.predict_qoi(Xs[te])
            z_true = sur.reduction.project(y)
            z_m, z_v = sur.predict(Xs[te])
            z_v = np.maximum(z_v, 1e-300)
            r = (z_true - z_m) / np.sqrt(z_v)
            nlpd = float(np.mean(0.5 * np.log(2 * np.pi * z_v) + 0.5 * r ** 2))
            ss_tot = np.sum((y - y.mean(axis=0)) ** 2)
            rows.append({"kernel": name, "case": cid,
                         "rel_rmse_pct": float(100 * np.sqrt(np.mean(((y - m) / y) ** 2))),
                         "q2": float(1 - np.sum((y - m) ** 2) / ss_tot) if ss_tot > 0 else np.nan,
                         "std_res_mean": float(r.mean()), "std_res_std": float(r.std()),
                         "coverage_95": float(np.mean(np.abs(r) <= 1.96)), "nlpd": nlpd,
                         "hit_bound": any(sur.hit_bound),
                         "learned_kernel": " | ".join(str(gp.kernel_) for gp in sur.gps)})
            arr["y"].append(y.ravel()); arr["m"].append(m.ravel()); arr["z"].append(r.ravel())
        arrays[name] = {k: np.concatenate(v) for k, v in arr.items()}
        if verbose:
            d = pd.DataFrame([r for r in rows if r["kernel"] == name])
            print(f"  {name:28s} rel. RMSE {d['rel_rmse_pct'].mean():.3f}%  Q2 {d['q2'].mean():.5f}  "
                  f"std res {d['std_res_std'].mean():.2f}  cov95 {d['coverage_95'].mean():.3f}  "
                  f"NLPD {d['nlpd'].mean():.3f}")
    per_case = pd.DataFrame(rows)
    summary = (per_case.groupby("kernel", sort=False)
               .agg(n_cases=("case", "count"), rel_rmse_pct_mean=("rel_rmse_pct", "mean"),
                    rel_rmse_pct_max=("rel_rmse_pct", "max"), q2_mean=("q2", "mean"),
                    std_res_mean=("std_res_mean", "mean"), std_res_std=("std_res_std", "mean"),
                    coverage_95=("coverage_95", "mean"), nlpd_mean=("nlpd", "mean"),
                    n_hit_bound=("hit_bound", "sum"))
               .reset_index())
    # Recommendation: lowest NLPD among the kernels whose residual std is within 0.8-1.25
    ok = summary[(summary["std_res_std"] >= 0.8) & (summary["std_res_std"] <= 1.25)]
    pick = (ok if len(ok) else summary).sort_values("nlpd_mean").iloc[0]["kernel"]
    summary["recommended"] = summary["kernel"] == pick
    return per_case, summary, arrays


def save_surrogates(sset: SurrogateSet, path: str | Path) -> Path:
    """Pickle the fitted surrogates (scikit-learn objects cannot go to HDF5).

    A pickle must be re-created if scikit-learn is updated.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(sset, f)
    return path


def load_surrogates(path: str | Path) -> SurrogateSet:
    with open(path, "rb") as f:
        return pickle.load(f)

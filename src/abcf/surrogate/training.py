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

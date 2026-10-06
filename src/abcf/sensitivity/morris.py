"""Morris elementary-effects screening (Morris, 1991, Technometrics 33(2),
161-174, doi:10.1080/00401706.1991.10484804), with the mu* measure of
Campolongo et al. (2007), Environmental Modelling & Software 22(10),
1509-1518, doi:10.1016/j.envsoft.2006.10.004.

Two ways of producing the model evaluations, same analysis:

* ``salib``: trajectories generated with SALib (Herman & Usher, 2017,
  JOSS 2(9), 97, doi:10.21105/joss.00097), model run through a
  ``ModelInterface``;
* ``dakota``: trajectories generated and run by Dakota (method
  ``psuade_moat``); the framework reads the Dakota tabular file.

The elementary effects are computed from consecutive points of each
trajectory, detecting which parameter changed, so the analysis does not
depend on the tool that produced the design.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def morris_design(bounds: dict, num_trajectories: int, num_levels: int, seed: int) -> pd.DataFrame:
    """Morris trajectories with SALib, in physical units. One row per model run."""
    from SALib.sample import morris as morris_sample
    problem = {"num_vars": len(bounds), "names": list(bounds),
               "bounds": [list(map(float, b)) for b in bounds.values()]}
    X = morris_sample.sample(problem, N=num_trajectories, num_levels=num_levels, seed=seed)
    return pd.DataFrame(X, columns=list(bounds))


def elementary_effects(X: np.ndarray, Y: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Elementary effects from trajectories.

    X: (r*(p+1), p) design in trajectory order; Y: (r*(p+1), n_out);
    bounds: (p, 2). Returns EE of shape (r, p, n_out), effects computed on
    inputs rescaled to [0, 1].
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).reshape(X.shape[0], -1)
    p = X.shape[1]
    if X.shape[0] % (p + 1):
        raise ValueError(f"The design has {X.shape[0]} rows, not a multiple of p+1 = {p + 1}: "
                         "it does not look like a Morris (one-at-a-time) design.")
    span = bounds[:, 1] - bounds[:, 0]
    U = (X - bounds[:, 0]) / span
    r = X.shape[0] // (p + 1)
    ee = np.full((r, p, Y.shape[1]), np.nan)
    for t in range(r):
        rows = slice(t * (p + 1), (t + 1) * (p + 1))
        Ut, Yt = U[rows], Y[rows]
        for j in range(p):
            dU = Ut[j + 1] - Ut[j]
            changed = np.flatnonzero(np.abs(dU) > 1e-12)
            if changed.size != 1:
                raise ValueError(f"Trajectory {t}, step {j}: {changed.size} parameters changed "
                                 "at once; a Morris design changes exactly one.")
            k = changed[0]
            ee[t, k] = (Yt[j + 1] - Yt[j]) / dU[k]
    return ee


def morris_indices(ee: np.ndarray, names: list) -> pd.DataFrame:
    """mu, mu* and sigma per parameter. For a vector output, mu* is
    reported per output and summarised by its maximum over the outputs."""
    mu = np.nanmean(ee, axis=0)                 # (p, n_out)
    mu_star = np.nanmean(np.abs(ee), axis=0)
    sigma = np.nanstd(ee, axis=0, ddof=1)
    df = pd.DataFrame({"parameter": names,
                       "mu_star": mu_star.max(axis=1),
                       "mu": mu[np.arange(len(names)), mu_star.argmax(axis=1)],
                       "sigma": sigma[np.arange(len(names)), mu_star.argmax(axis=1)],
                       "output_index_of_max": mu_star.argmax(axis=1)})
    df["mu_star_normalised"] = df["mu_star"] / df["mu_star"].max()
    return df.sort_values("mu_star", ascending=False).reset_index(drop=True)


def run_morris_salib(model, case_id: str, bounds: dict, cfg_m: dict, run_dir: Path,
                     seed: int, progress: bool = True):
    """Generate the design, run the model through ``model.run`` and analyse."""
    from tqdm import tqdm
    design = morris_design(bounds, cfg_m["num_trajectories"], cfg_m["num_levels"], seed)
    outputs = []
    for i, row in tqdm(design.iterrows(), total=len(design), desc=f"Morris {case_id}",
                       disable=not progress):
        outputs.append(model.run(row.to_dict(), case_id, Path(run_dir) / case_id / f"eval_{i:05d}"))
    Y = np.vstack(outputs)
    ee = elementary_effects(design.to_numpy(), Y, np.array(list(bounds.values()), dtype=float))
    return morris_indices(ee, list(bounds)), design, Y


def read_dakota_tabular(path: str | Path, param_names: list, response_names: list) -> tuple:
    """Read a Dakota tabular file (``tabular_data`` output, annotated format).

    Returns (X, Y) in run order. Lines are whitespace separated; the
    header starts with '%eval_id'.
    """
    df = pd.read_csv(path, sep=r"\s+")
    df.columns = [c.lstrip("%") for c in df.columns]
    missing = [c for c in param_names + response_names if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: columns not found in the Dakota tabular file: {missing}")
    df = df.sort_values("eval_id")
    return df[param_names].to_numpy(dtype=float), df[response_names].to_numpy(dtype=float)


def run_morris_dakota(tabular_path, bounds: dict, response_names: list):
    """Analyse a Morris design run by Dakota (method psuade_moat)."""
    X, Y = read_dakota_tabular(tabular_path, list(bounds), response_names)
    ee = elementary_effects(X, Y, np.array(list(bounds.values()), dtype=float))
    return morris_indices(ee, list(bounds)), X, Y


def plot_morris(table: pd.DataFrame, title: str, path) -> None:
    """mu* (x) against sigma (y), one labelled point per parameter.

    Dashed lines sigma/mu* = 0.1, 0.5 and 1 separate (almost) linear
    effects, monotonic effects, and non-linear effects or interactions
    (Garcia Sanchez et al., 2014, Energy and Buildings 68, 741-750,
    doi:10.1016/j.enbuild.2012.08.048).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 5))
    x, y = table["mu_star"].to_numpy(), table["sigma"].to_numpy()
    top = 1.15 * max(x.max(), y.max(), 1e-12)
    for ratio, style in ((0.1, ":"), (0.5, "--"), (1.0, "-.")):
        ax.plot([0, top], [0, ratio * top], color="gray", ls=style, lw=1,
                label=rf"$\sigma/\mu^*$ = {ratio:g}")
    ax.scatter(x, y, s=50, color="#0072b2", zorder=3)
    for xi, yi, name in zip(x, y, table["parameter"]):
        ax.annotate(name, (xi, yi), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel(r"$\mu^*$ (mean of |elementary effects|)")
    ax.set_ylabel(r"$\sigma$ (std of elementary effects)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

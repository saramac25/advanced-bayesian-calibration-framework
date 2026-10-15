"""Sobol' indices of the 5 TRANSURANUS multipliers (the parameters retained by Robertson et al.
and varied in the Dakota runs) on the PCA scores of the centreline temperature history of
every case, i.e. on the quantities the GPs emulate and the likelihood uses (GP mean as model).

For each case and principal component k: S1_k, ST_k (Saltelli scheme, Jansen estimator,
SALib). Since the components are orthogonal, the variance of the whole temperature history
explained by the retained components is the sum of the variances of the scores, and the
index of the history is the variance-weighted mean of the per-component indices
(generalised sensitivity index: Lamboni, Monod and Makowski 2011, Reliab. Eng. Syst. Saf.
96(4) 450-459; Gamboa et al. 2014, Electron. J. Stat. 8 575-603):
    GST = sum_k Var(z_k) ST_k / sum_k Var(z_k)       (likewise GS1)
A parameter acting only on a minor component (e.g. the late, burnup-dependent shape) gets a
large ST_k there but a small GST: both are reported; the decision uses GST.

To read the indices in kelvin (with --data, the framework input file): the components are
orthonormal, so sum_k Var(z_k) is the sum over the n_t points of the variance of the
temperature, and the root-mean-square (over time) standard deviation of the temperature
due to parameter i, with its interactions, is
    sigma_i = sqrt(GST_i sum_k Var(z_k) / n_t)        [K]
compared with the measurement uncertainty sigma_exp = rel_noise x RMS(measurement) [K].
sigma_i / sigma_exp < 1: the effect of the parameter is below the measurement uncertainty
and the data can hardly inform it.

No literature uncertainty is available for these multipliers: the input distributions are
uniform over the ranges of the Dakota training runs (gap conductance [0.8, 1.2], the other
multipliers [0.5, 1.5]), i.e. the same prior as the calibration. ST >= --threshold ->
"calibrate", otherwise the parameter can be fixed at the nominal value.

    python sobol_tu.py --surrogates results/common/surrogates/surrogates.pkl --out results/common/sobol
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

import pandas as pd  # noqa: E402

from abcf.sensitivity.sobol import _sample, sobol_summary  # noqa: E402
from abcf.surrogate import load_surrogates  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--surrogates", default="results/common/surrogates/surrogates.pkl")
    ap.add_argument("--out", default="results/common/sobol")
    ap.add_argument("--config", default=str(HERE / "config_tu.yaml"))
    ap.add_argument("--n-base", type=int, default=1024)
    ap.add_argument("--threshold", type=float, default=0.01)
    ap.add_argument("--data", default=None, help="input_data.h5: n_t and measurements, for sigma_i in K")
    ap.add_argument("--rel-noise", type=float, default=0.01, help="relative measurement uncertainty")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sset = load_surrogates(args.surrogates)
    cfg = yaml.safe_load(Path(args.config).read_text())
    labels = cfg.get("labels", {})
    calibrated = [p for p, s in cfg["parameters"].items() if s.get("calibrate")]
    lo = sset.scaler.inverse(sset.scaler.lower)
    hi = sset.scaler.inverse(sset.scaler.upper)
    dists = {p: {"type": "uniform", "low": float(a), "high": float(b)}
             for p, a, b in zip(sset.param_names, lo, hi)}

    from SALib.analyze import sobol as sobol_analyze
    names = list(sset.param_names)
    problem, X = _sample(dists, names, lo, hi, args.n_base, seed=1)
    Xs = sset.scaler.transform(X)
    per_pc, rows = [], []
    for cid in sset.surrogates:
        Z = np.asarray(sset[cid].predict(Xs, return_var=False)).reshape(X.shape[0], -1)
        var = Z.var(axis=0)
        s1, st = np.zeros((Z.shape[1], len(names))), np.zeros((Z.shape[1], len(names)))
        for k in range(Z.shape[1]):
            r = sobol_analyze.analyze(problem, Z[:, k], calc_second_order=False, num_resamples=100, seed=1)
            s1[k], st[k] = r["S1"], r["ST"]
            for i, p in enumerate(names):
                per_pc.append({"case": cid, "component": f"PC{k + 1}", "variance_share": var[k] / var.sum(),
                               "parameter": p, "S1": r["S1"][i], "ST": r["ST"][i]})
        w = var / var.sum()
        for i, p in enumerate(names):          # generalised indices of the whole history
            rows.append({"case": cid, "parameter": p, "S1": float(w @ s1[:, i]), "ST": float(w @ st[:, i]),
                         "var_scores_sum": float(var.sum()),
                         "n_components": Z.shape[1]})
    pd.DataFrame(per_pc).to_csv(out / "sobol_per_case_and_component.csv", index=False)
    table = pd.DataFrame(rows)
    if args.data:
        from abcf.io import load_input_data
        d = load_input_data(args.data, "all")
        n_t = {c: d.cases[c].meas.size for c in d.cases}
        s_exp = {c: args.rel_noise * float(np.sqrt(np.mean(d.cases[c].meas ** 2))) for c in d.cases}
        table["sigma_i_K"] = np.sqrt(table["ST"].clip(lower=0) * table["var_scores_sum"] / table["case"].map(n_t))
        table["sigma_exp_K"] = table["case"].map(s_exp)
        table["ratio_to_sigma_exp"] = table["sigma_i_K"] / table["sigma_exp_K"]
    table.to_csv(out / "sobol_per_case.csv", index=False)
    summ = sobol_summary(table, args.threshold)
    if "sigma_i_K" in table:
        g = table.groupby("parameter")
        summ["sigma_i_K_mean"] = g["sigma_i_K"].mean()
        summ["sigma_exp_K_mean"] = g["sigma_exp_K"].mean()
        summ["ratio_mean"] = g["ratio_to_sigma_exp"].mean()
        summ["ratio_q97_5"] = g["ratio_to_sigma_exp"].quantile(0.975)
        summ["n_cases_ratio_ge_1"] = g["ratio_to_sigma_exp"].apply(lambda r: int((r >= 1).sum()))
    summ["calibrated_in_config"] = summ.index.isin(calibrated)
    summ.to_csv(out / "sobol_summary.csv")
    print(f"Generalised Sobol' indices (PCA scores, variance-weighted) on {table['case'].nunique()} cases, "
          f"uniform inputs over the training ranges, {args.n_base} base samples")
    cols = ["S1_mean", "ST_mean", "ST_q2_5", "ST_q97_5", "suggestion", "calibrated_in_config"]
    print(summ[cols].to_string(float_format="%.4f"))
    if "sigma_i_K_mean" in summ:
        print("\nIn kelvin (RMS over time): sigma_i = sqrt(GST Var), vs measurement uncertainty sigma_exp")
        print(summ[["sigma_i_K_mean", "sigma_exp_K_mean", "ratio_mean", "ratio_q97_5", "n_cases_ratio_ge_1"]]
              .to_string(float_format="%.3f"))

    order = list(summ.index)[::-1]
    floor = 1e-4
    fig, ax = plt.subplots(figsize=(8, 0.75 * len(order) + 1.8))
    for i, p in enumerate(order):
        r = summ.loc[p]
        col = "#0072b2" if p in calibrated else "#9a9a9a"
        st, s1 = max(r["ST_mean"], floor), max(r["S1_mean"], floor)
        ax.barh(i + 0.18, st, height=0.34, color=col)
        ax.barh(i - 0.18, s1, height=0.34, facecolor="white", edgecolor=col, hatch="///")
        ax.errorbar(st, i + 0.18, xerr=[[st - max(r["ST_q2_5"], floor)], [max(r["ST_q97_5"], floor) - st]],
                    fmt="none", ecolor="#222222", capsize=3)
    ax.axvline(args.threshold, color="#d55e00", ls="--")
    ax.set_yticks(np.arange(len(order)), [labels.get(p, p) for p in order])
    ax.set_xscale("log")
    ax.set_xlim(floor, 3)
    ax.set_xlabel("Generalised Sobol' index (variance of the temperature history, PCA scores)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"sobol_tu.{ext}", dpi=200)
    print(f"Written {out}/sobol_per_case_and_component.csv, sobol_per_case.csv, sobol_summary.csv, sobol_tu.png/.pdf")


if __name__ == "__main__":
    main()

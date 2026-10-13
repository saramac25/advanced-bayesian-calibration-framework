"""Morris screening of the 17 OFFBEAT multipliers on the FCT at EOL of the 130 TP3 rods (Chapter 5).

Input: the Dakota tabular file of the psuade_moat run (10 trajectories x 18 points = 180
evaluations, 4 levels in [0.5, 1.5]); a partial file extracted from dakota.rst
(dakota_restart_util to_tabular) is accepted: only the completed evaluations are used.

For every rod and parameter: elementary effects EE = dFCT / dx (x rescaled to [0, 1], i.e. per
unit of the [0.5, 1.5] range) from the consecutive points of each trajectory, only when both
points exist and both runs reached the end of life (NaN = run stopped early, excluded for that
rod only). mu* = mean |EE| (Campolongo et al. 2007), sigma = std of EE (Morris 1991).
Across the rods: mean and 2.5-97.5 percentile interval of mu* and sigma.

    python morris_ch5.py inputs/morris_tabular.dat [--out results/morris]

Outputs: morris_per_rod.csv (rod, parameter, mu_star, mu, sigma, n_ee), morris_summary.csv
(per parameter: mean and percentiles across rods, EEs used / possible), morris_ch5.png/.pdf.

References: Morris (1991), Technometrics 33(2) 161-174, doi:10.1080/00401706.1991.10484804;
Campolongo, Cariboni, Saltelli (2007), Environ. Model. Softw. 22(10) 1509-1518,
doi:10.1016/j.envsoft.2006.10.004.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PARAMS = ["UO2_rho", "UO2_Cp", "UO2_k", "UO2_emissivity", "UO2_E", "UO2_nu", "UO2_alphaT",
          "Zirc_rho", "Zirc_Cp", "Zirc_k", "Zirc_emissivity", "Zirc_E", "Zirc_nu", "Zirc_alphaT",
          "UO2_epsilonDensification", "UO2_epsilonRelocation", "UO2_epsilonSwelling"]
LABEL = {"UO2_rho": r"UO$_2$ $\rho$", "UO2_Cp": r"UO$_2$ $C_p$", "UO2_k": r"UO$_2$ $k$",
         "UO2_emissivity": r"UO$_2$ $\varepsilon$", "UO2_E": r"UO$_2$ $E$", "UO2_nu": r"UO$_2$ $\nu$",
         "UO2_alphaT": r"UO$_2$ $\alpha_T$", "Zirc_rho": r"Zry $\rho$", "Zirc_Cp": r"Zry $C_p$",
         "Zirc_k": r"Zry $k$", "Zirc_emissivity": r"Zry $\varepsilon$", "Zirc_E": r"Zry $E$",
         "Zirc_nu": r"Zry $\nu$", "Zirc_alphaT": r"Zry $\alpha_T$",
         "UO2_epsilonDensification": r"UO$_2$ densification",
         "UO2_epsilonRelocation": r"UO$_2$ relocation", "UO2_epsilonSwelling": r"UO$_2$ swelling"}
LOW, HIGH, N_TRAJ = 0.5, 1.5, 10
# label offsets (points) for parameters whose markers are close to each other
LABEL_OFFSET = {"UO2_alphaT": (4, -17, "center"), "UO2_epsilonRelocation": (10, -15, "left"),
                "Zirc_alphaT": (-9, 6, "right"), "Zirc_k": (2, 9, "center"),
                "UO2_epsilonSwelling": (8, 8, "left")}
CALIBRATION_CANDIDATES = ["UO2_rho", "UO2_k", "UO2_alphaT", "UO2_epsilonRelocation", "UO2_epsilonSwelling"]


def read_tabular(path):
    df = pd.read_csv(path, sep=r"\s+")
    df.columns = [c.lstrip("%") for c in df.columns]
    df = df.drop_duplicates("eval_id").set_index("eval_id").sort_index()
    rods = [c for c in df.columns if c.startswith("ROD")]
    return df, sorted(rods, key=lambda r: int(r[3:]))


def elementary_effects(df, rods):
    """EE[traj, param, rod] (NaN where not available) and the step table."""
    p = len(PARAMS)
    ee = np.full((N_TRAJ, p, len(rods)), np.nan)
    steps = []
    for t in range(N_TRAJ):
        for j in range(p):
            a, b = t * (p + 1) + j + 1, t * (p + 1) + j + 2          # Dakota eval ids (1-based)
            if a not in df.index or b not in df.index:
                steps.append((t, j, None, "evaluation not completed"))
                continue
            dU = (df.loc[b, PARAMS].to_numpy(float) - df.loc[a, PARAMS].to_numpy(float)) / (HIGH - LOW)
            ch = np.flatnonzero(np.abs(dU) > 1e-9)
            if ch.size != 1:
                raise ValueError(f"evals {a}-{b}: {ch.size} parameters changed (not a Morris step)")
            k = ch[0]
            ee[t, k] = (df.loc[b, rods].to_numpy(float) - df.loc[a, rods].to_numpy(float)) / dU[k]
            steps.append((t, j, PARAMS[k], "ok"))
    return ee, steps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tabular")
    ap.add_argument("--out", default="results/morris")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df, rods = read_tabular(args.tabular)
    ee, steps = elementary_effects(df, rods)
    n_ok = int(np.isfinite(ee).sum())
    n_possible = N_TRAJ * len(PARAMS) * len(rods)
    print(f"{len(df)} of {N_TRAJ * (len(PARAMS) + 1)} evaluations completed; "
          f"elementary effects available: {n_ok} of {n_possible} ({100 * n_ok / n_possible:.1f}%)")
    n_nan_runs = int(df[rods].isna().to_numpy().sum())
    print(f"rod runs stopped before the end of life in the completed evaluations: {n_nan_runs} of "
          f"{df[rods].size} ({100 * n_nan_runs / df[rods].size:.2f}%)")

    rows = []
    for i, rod in enumerate(rods):
        for k, prm in enumerate(PARAMS):
            e = ee[:, k, i]
            e = e[np.isfinite(e)]
            rows.append({"rod": rod, "parameter": prm, "n_ee": len(e),
                         "mu_star": np.mean(np.abs(e)) if len(e) else np.nan,
                         "mu": np.mean(e) if len(e) else np.nan,
                         "sigma": np.std(e, ddof=1) if len(e) > 1 else np.nan})
    per = pd.DataFrame(rows)
    per.to_csv(out / "morris_per_rod.csv", index=False)

    def q(s, a):
        return np.nanpercentile(s, a)
    summ = (per.groupby("parameter")
            .agg(mu_star_mean=("mu_star", "mean"), mu_star_q2_5=("mu_star", lambda s: q(s, 2.5)),
                 mu_star_q97_5=("mu_star", lambda s: q(s, 97.5)),
                 sigma_mean=("sigma", "mean"), sigma_q2_5=("sigma", lambda s: q(s, 2.5)),
                 sigma_q97_5=("sigma", lambda s: q(s, 97.5)),
                 mu_mean=("mu", "mean"), ee_used=("n_ee", "sum"), ee_min_per_rod=("n_ee", "min"))
            .reindex(PARAMS))
    summ["ee_possible"] = N_TRAJ * len(rods)
    summ["mu_star_rank"] = summ["mu_star_mean"].rank(ascending=False).astype(int)
    summ = summ.sort_values("mu_star_mean", ascending=False)
    summ.to_csv(out / "morris_summary.csv")
    pd.set_option("display.width", 200)
    print(summ[["mu_star_mean", "mu_star_q2_5", "mu_star_q97_5", "sigma_mean", "mu_mean",
                "ee_used", "ee_possible"]].to_string(float_format="%.2f"))

    # ---- figure: mu* vs sigma, mean over the rods with 95% percentile intervals -------------
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    s = summ.reindex(PARAMS)
    x, y = s["mu_star_mean"].to_numpy(), s["sigma_mean"].to_numpy()
    xerr = [x - s["mu_star_q2_5"].to_numpy(), s["mu_star_q97_5"].to_numpy() - x]
    yerr = [y - s["sigma_q2_5"].to_numpy(), s["sigma_q97_5"].to_numpy() - y]
    cand = np.array([p in CALIBRATION_CANDIDATES for p in PARAMS])
    for m, col, lab in ((cand, "#0072b2", "parameters of the GPs (LHS)"),
                        (~cand, "#7f7f7f", "other OFFBEAT parameters")):
        ax.errorbar(x[m], y[m], xerr=[e[m] for e in xerr], yerr=[e[m] for e in yerr], fmt="o",
                    color=col, ms=7, elinewidth=1.2, capsize=3, label=lab, zorder=3)
    lo = 0.05                                    # [K]: parameters below are negligible
    hi = max(np.nanmax(s["mu_star_q97_5"]), np.nanmax(s["sigma_q97_5"])) * 2
    sig_exp = 0.01 * float(np.nanmean(df[rods].to_numpy(float)))   # 1% of the mean FCT at EOL
    ax.axvline(sig_exp, color="#d55e00", lw=1.2, ls="--", zorder=1,
               label=rf"1% of the mean FCT at EOL ($\approx$ measurement uncertainty, {sig_exp:.0f} K)")
    for ratio, ls in ((0.1, ":"), (0.5, "--"), (1.0, "-.")):
        ax.plot([lo, hi], [ratio * lo, ratio * hi], color="0.6", ls=ls, lw=1,
                label=rf"$\sigma/\mu^* = {ratio:g}$")
    # labels: one per parameter, placed alternately above/below the point to limit overlaps;
    # parameters with mu* < 1 K are grouped in one note (negligible on the FCT at EOL)
    order = np.argsort(-np.nan_to_num(x))
    small = []
    for n, i in enumerate(order):
        prm = PARAMS[i]
        if not np.isfinite(x[i]):
            continue
        if x[i] < 1.0:
            small.append(LABEL[prm])
            continue
        dx, dy, ha = LABEL_OFFSET.get(prm, (7, 8 if n % 2 == 0 else -14, "left"))
        ax.annotate(LABEL[prm], (x[i], y[i]), textcoords="offset points", xytext=(dx, dy), fontsize=9,
                    ha=ha, color="#0b2c5a" if prm in CALIBRATION_CANDIDATES else "0.3")
    if small:
        ax.text(0.98, 0.03, r"$\mu^* < 1$ K: " + ", ".join(small), transform=ax.transAxes, ha="right",
                va="bottom", fontsize=8, color="0.35", wrap=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"$\mu^*$, mean $|$elementary effect$|$ of the FCT at EOL [K]")
    ax.set_ylabel(r"$\sigma$, std of the elementary effects [K]")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(alpha=0.25, which="both", lw=0.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"morris_ch5.{ext}", dpi=200)
    print(f"\nWritten {out}/morris_per_rod.csv, morris_summary.csv, morris_ch5.png/.pdf")


if __name__ == "__main__":
    main()

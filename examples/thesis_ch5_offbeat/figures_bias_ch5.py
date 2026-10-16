"""Chapter 5, model-bias study: figures of the two cases with k calibrated
(modelbias_uniform_k: bias common to all rods; modelbias_struct_k: bias dependent on the rod
power), from the 100 repetitions (repeat_ch5.py) and from the reference runs.

    python figures_bias_ch5.py [--results results] [--out results/figures]

bias_coverage_ch5   95% interval of k in each repetition (new 1% measurement noise each time):
                    standard (SC) and hierarchical population of a new rod (mu + sigma z);
                    number of repetitions covering the true value in the legend.
bias_rodlevel_ch5   reference runs: posterior of the rod-level k_i of the hierarchical
                    calibration against the time-averaged LHGR of the rod, with the 95% interval
                    of the standard calibration and of the hierarchical population. A trend of
                    k_i with the power reveals a condition-dependent model error (the true k is
                    the same for all rods).
bias_coverage_ch5.csv  coverage table of the two cases (SC, HC mu, HC population).

The power of the rods is recovered from inputs/fixed_values_modelbias_struct_k.csv
(rho_i = 1 + b_rho (P_i - P_ref), make_modelbias_struct.py).
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from make_modelbias_struct import B_RHO, P_REF  # noqa: E402

HERE = Path(__file__).resolve().parent
C_SC, C_HC, C_TRUE = "#7b3294", "#e69f00", "#0072b2"
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 13,
                     "axes.titlesize": 14, "axes.labelsize": 14, "legend.fontsize": 11})
CASES = [("modelbias_uniform_k", "Model bias common to all rods"),
         ("modelbias_struct_k", "Model bias dependent on the rod power")]


def rod_power() -> pd.Series:
    f = pd.read_csv(HERE / "inputs" / "fixed_values_modelbias_struct_k.csv").set_index("case")
    return P_REF + (f["UO2_rho"] - 1.0) / B_RHO


def coverage(res: Path, out: Path, true_k: float):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    rows = []
    for ax, (case, title) in zip(axes, CASES):
        d = pd.read_csv(res / f"repeat_{case}" / "repetitions.csv").sort_values("rep")
        r = d["rep"].to_numpy()
        for tag, col, off, lab in (("SC", C_SC, -0.18, "Standard"),
                                   ("HC_pop", C_HC, 0.18, "Hierarchical (population)")):
            cov = int(d[f"{tag}_UO2_k_cov"].sum())
            ax.vlines(r + off, d[f"{tag}_UO2_k_lo"], d[f"{tag}_UO2_k_hi"], color=col, lw=1.4,
                      alpha=0.85, label=f"{lab}: {cov}/{len(d)} cover")
        for tag in ("SC", "HC_mu", "HC_pop"):
            rows.append({"case": case, "estimate": tag, "covered": int(d[f"{tag}_UO2_k_cov"].sum()),
                         "n": len(d), "mean_of_means": d[f"{tag}_UO2_k_mean"].mean(),
                         "mean_width": (d[f"{tag}_UO2_k_hi"] - d[f"{tag}_UO2_k_lo"]).mean(),
                         "sigma_k_mean": d["HC_sigma_UO2_k_mean"].mean()})
        ax.axhline(true_k, color=C_TRUE, lw=2, label="True value")
        ax.set_title(title)
        ax.set_xlabel("Repetition")
        ax.set_xlim(-2, len(d) + 1)
        ax.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="none")
    lo = min(ax.get_ylim()[0] for ax in axes)
    hi = max(ax.get_ylim()[1] for ax in axes)
    axes[0].set_ylim(lo, hi + 0.35 * (hi - lo))           # room for the legends above the intervals
    axes[0].set_ylabel("$k$ multiplier, 95% interval")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"bias_coverage_ch5.{ext}", dpi=200)
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "bias_coverage_ch5.csv", index=False)
    print(tab.round(4).to_string(index=False))


def rodlevel(res: Path, out: Path, true_k: float):
    p = rod_power()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (case, title) in zip(axes, CASES):
        s = pd.read_csv(res / case / "postprocessing" / "posterior_summary.csv")
        x = s[(s.method == "HC") & s.block.str.startswith("X[")].copy()
        x["rod"] = x.block.str[2:-1]
        x["P"] = x.rod.map(p)
        sc = s[(s.method == "SC") & (s.block == "theta")].iloc[0]
        pop = s[(s.method == "HC") & (s.block == "population")].iloc[0]
        ax.axhspan(sc["q2.5"], sc["q97.5"], color=C_SC, alpha=0.18, label="Standard, 95%")
        ax.axhspan(pop["q2.5"], pop["q97.5"], color=C_HC, alpha=0.12, label="Hierarchical population, 95%")
        ax.errorbar(x.P, x["mean"], yerr=[x["mean"] - x["q2.5"], x["q97.5"] - x["mean"]], fmt="o",
                    color=C_HC, ecolor=C_HC, capsize=3, label="Hierarchical, rod-level $k_i$")
        b = np.polyfit(x.P, x["mean"], 1)
        r = np.corrcoef(x.P, x["mean"])[0, 1]
        pp = np.linspace(x.P.min() - 0.2, x.P.max() + 0.2, 10)
        ax.plot(pp, np.polyval(b, pp), "--", color="0.35", lw=1.2,
                label=f"linear trend: {b[0]:+.4f} per kW/m ($r$ = {r:.2f})")
        ax.axhline(true_k, color=C_TRUE, lw=2, label="True value")
        ax.set_title(title)
        ax.set_xlabel("Time-averaged linear heat rate [kW/m]")
        ax.legend(frameon=False, fontsize=9.5, loc="lower left")
        print(f"{case}: rod-level k vs power slope {b[0]:+.4f} per kW/m, r = {r:.2f}")
    axes[0].set_ylabel("$k$ multiplier")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"bias_rodlevel_ch5.{ext}", dpi=200)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--out", default=None, help="default: <results>/figures")
    a = ap.parse_args()
    res = Path(a.results)
    out = Path(a.out) if a.out else res / "figures"
    out.mkdir(parents=True, exist_ok=True)
    true_k = json.loads((HERE / "inputs" / "metadata.json").read_text())["true_A"]["UO2_k"]
    coverage(res, out, true_k)
    rodlevel(res, out, true_k)

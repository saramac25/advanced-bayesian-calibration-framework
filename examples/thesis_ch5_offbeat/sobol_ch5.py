"""Sobol' indices of the 5 GP parameters on the FCT at EOL of every rod (Chapter 5).

Decides which of the 5 parameters of the GPs are worth calibrating and which can be fixed
(or, in the model-bias case, left at a wrong value): a parameter whose total index ST
(fraction of the variance of the FCT explained by the parameter and its interactions) is
below the threshold can be fixed without changing the FCT appreciably.

Input uncertainty = the prior knowledge before the calibration: centred on the OFFBEAT
default (1.0) with the widths of the literature (inputs/metadata.json: k, alphaT, swelling
normal, relocation uniform, density normal; Fiorina et al. 2022, Ikonen 2016), truncated
to the GP training range [0.5, 1.5]. Model = GP mean of each rod (abcf.sensitivity.sobol).

    python sobol_ch5.py --surrogates results/common/surrogates/surrogates.pkl --out results/common/sobol

Outputs: sobol_per_rod.csv, sobol_summary.csv (mean and 2.5-97.5 percentiles across the rods,
suggestion calibrate / fix), sobol_ch5.png/.pdf.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from abcf.sensitivity.sobol import sobol_indices, sobol_summary  # noqa: E402
from abcf.surrogate import load_surrogates  # noqa: E402

HERE = Path(__file__).resolve().parent
C_CAL, C_FIX, INK = "#0072b2", "#9a9a9a", "#222222"


def prior_distributions(meta: dict, centre: str = "nominal") -> dict:
    out = {}
    for p, d in meta["distributions"].items():
        m = 1.0 if centre == "nominal" else float(d["mean"])
        if d["type"] == "normal":
            out[p] = {"type": "normal", "mean": m, "std": float(d["std"])}
        else:
            h = float(d["sigma_or_half_range"])
            out[p] = {"type": "uniform", "low": m - h, "high": m + h}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--surrogates", default="results/common/surrogates/surrogates.pkl")
    ap.add_argument("--out", default="results/common/sobol")
    ap.add_argument("--n-base", type=int, default=1024)
    ap.add_argument("--threshold", type=float, default=0.01, help="ST below this -> the parameter can be fixed")
    ap.add_argument("--centre", choices=["nominal", "true"], default="nominal",
                    help="centre of the input distributions: OFFBEAT default (prior knowledge) or true centres")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sset = load_surrogates(args.surrogates)
    meta = json.loads((HERE / "inputs" / "metadata.json").read_text())
    cfg = yaml.safe_load((HERE / "config_epistemic.yaml").read_text())
    labels = cfg.get("labels", {})
    calibrated = [p for p, s in cfg["parameters"].items() if s.get("calibrate")]
    dists = prior_distributions(meta, args.centre)

    table = sobol_indices(sset, dists, n_base=args.n_base)
    table.to_csv(out / "sobol_per_rod.csv", index=False)
    summ = sobol_summary(table, args.threshold)
    summ["calibrated_in_this_study"] = summ.index.isin(calibrated)
    summ.to_csv(out / "sobol_summary.csv")
    print(f"Sobol indices on {table['case'].nunique()} rods, {args.n_base} base samples, "
          f"input distributions centred on the {'OFFBEAT default' if args.centre == 'nominal' else 'true centres'}")
    print(summ[["S1_mean", "ST_mean", "ST_q2_5", "ST_q97_5", "suggestion", "calibrated_in_this_study"]]
          .to_string(float_format="%.4f"))

    # ---- figure: S1 and ST per parameter (mean over the rods, 2.5-97.5% across rods) --------
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False})
    order = list(summ.index)[::-1]                      # largest ST on top
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8.5, 0.75 * len(order) + 1.8))
    floor = 1e-4
    for i, p in enumerate(order):
        r = summ.loc[p]
        col = C_CAL if p in calibrated else C_FIX
        st, s1 = max(r["ST_mean"], floor), max(r["S1_mean"], floor)
        ax.barh(i + 0.18, st, height=0.34, color=col, alpha=0.9, zorder=3)
        ax.barh(i - 0.18, s1, height=0.34, facecolor="white", edgecolor=col, lw=1.6, hatch="///", zorder=3)
        ax.errorbar(st, i + 0.18, xerr=[[st - max(r["ST_q2_5"], floor)], [max(r["ST_q97_5"], floor) - st]],
                    fmt="none", ecolor=INK, elinewidth=1, capsize=3, zorder=4)
        ax.text(max(st, s1) * 1.6, i, f"ST = {r['ST_mean']:.3g}", va="center", fontsize=9, color=INK)
    ax.axvline(args.threshold, color="#d55e00", ls="--", lw=1.3, zorder=2)
    ax.text(args.threshold * 1.08, len(order) - 0.45, f"threshold ST = {args.threshold:g}",
            color="#d55e00", fontsize=9, va="bottom")
    ax.set_yticks(y, [labels.get(p, p) for p in order])
    for t, p in zip(ax.get_yticklabels(), order):
        t.set_fontweight("bold" if p in calibrated else "normal")
        t.set_color(C_CAL if p in calibrated else "0.35")
    ax.set_xscale("log")
    ax.set_xlim(floor, 3)
    ax.set_xlabel("Sobol' index (fraction of the variance of the FCT at EOL)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_CAL, label="ST, calibrated parameter"),
                       Patch(color=C_FIX, label="ST, fixed parameter"),
                       Patch(facecolor="white", edgecolor=INK, hatch="///", label="S1 (first order)")],
              frameon=False, fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.25, which="both", lw=0.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"sobol_ch5.{ext}", dpi=200)
    print(f"Written {out}/sobol_per_rod.csv, sobol_summary.csv, sobol_ch5.png/.pdf")


if __name__ == "__main__":
    main()

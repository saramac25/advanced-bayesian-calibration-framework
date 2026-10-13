"""Illustration of the PCA reduction of a time-dependent QoI for one case (Chapter 4 figure).

For one case (e.g. a fuel rod with 3 retained components) the figure shows, from the
training runs of the input file:

  (a) the original time series of all runs, their mean and one highlighted run;
  (b) the cumulative explained variance and the number K of retained components;
  (c) the K components (principal directions) as functions of time;
  (d) the reconstruction of the highlighted run with 1, ..., K components,
      y(t) = mean(t) + sum_k z_k phi_k(t).
With --all-panels, also the scores of the runs (z1-z2 scatter coloured by a parameter, and
their histograms): 6 panels.

    python tools/plot_pca_illustration.py data/input_data.h5 --case ROD12 \
        --variance 0.99 --color-by UO2_k --out pca_illustration --qoi "FCT [K]" --time-unit d

Without --case, the first case with the largest number of retained components is used.
Writes <out>.png and <out>.pdf. PCA as in the framework (abcf.surrogate.gp.OutputReduction:
scikit-learn PCA on the centred outputs, K = smallest number with cumulative explained
variance >= variance).
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from abcf.io import load_input_data  # noqa: E402
from abcf.surrogate.gp import OutputReduction  # noqa: E402

COLORS = ["#0072b2", "#e69f00", "#009e73", "#cc79a7", "#56b4e9"]   # fixed order, component k
STYLES = ["-", "--", "-.", ":", (0, (5, 1))]
INK, MUTED = "#222222", "#9a9a9a"
TIME_SCALE = {"s": 1.0, "h": 3600.0, "d": 86400.0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_file")
    ap.add_argument("--case", default=None)
    ap.add_argument("--variance", type=float, default=0.99)
    ap.add_argument("--color-by", default=None, help="parameter used to colour the scores (default: first)")
    ap.add_argument("--run", type=int, default=None, help="index of the highlighted run (default: median score)")
    ap.add_argument("--qoi", default="QoI")
    ap.add_argument("--time-unit", default="s", choices=list(TIME_SCALE))
    ap.add_argument("--out", default="pca_illustration")
    ap.add_argument("--all-panels", action="store_true",
                    help="also the scores (scatter z1-z2 and histograms): 6 panels instead of 4")
    args = ap.parse_args()

    data = load_input_data(args.input_file)
    cases = {c: v for c, v in data.cases.items() if not v.is_scalar}
    if not cases:
        raise SystemExit("No time-dependent case in the input file.")
    ncomp = {c: OutputReduction(v.pred, args.variance).n_components for c, v in cases.items()}
    cid = args.case or max(cases, key=lambda c: (ncomp[c], -list(cases).index(c)))
    case = cases[cid]
    Y = case.pred[~np.isnan(case.pred).any(axis=1)]
    ids = np.asarray(case.eval_ids)[~np.isnan(case.pred).any(axis=1)]
    t = np.asarray(case.tim, dtype=float) / TIME_SCALE[args.time_unit]

    red = OutputReduction(Y, args.variance)
    K, W, mu = red.n_components, red.W, red.center
    Z = red.project(Y)
    cum = np.cumsum(PCA().fit(Y).explained_variance_ratio_)
    # default run: the one whose smallest standardised |score| is largest, so that every
    # component visibly changes its reconstruction in panel (f)
    run = args.run if args.run is not None else int(np.argmax(np.min(np.abs(Z / Z.std(axis=0)), axis=1)))
    par = args.color_by or data.inputs.columns[0]
    cval = data.inputs.loc[ids, par].to_numpy(dtype=float)

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    keys = ["a", "b", "c", "d", "e", "f"] if args.all_panels else ["a", "b", "c", "f"]
    fig, grid = plt.subplots(2, 3 if args.all_panels else 2,
                             figsize=(15, 8.6) if args.all_panels else (11.5, 8.6))
    AX = dict(zip(keys, grid.flat))
    lab = {k: f"({chr(97 + i)})" for i, k in enumerate(keys)}      # panel letters in reading order
    tl = f"time [{args.time_unit}]"

    a = AX["a"]
    for y in Y:
        a.plot(t, y, color=MUTED, lw=0.4, alpha=0.35)
    a.plot(t, mu, color=INK, lw=2, label="mean of the runs")
    a.plot(t, Y[run], color=COLORS[0], lw=2, label=f"highlighted run (eval {ids[run]})")
    a.set(xlabel=tl, ylabel=args.qoi, title=f"{lab['a']} {len(Y)} training runs of {cid}")
    a.legend(frameon=False, fontsize=9)

    a = AX["b"]
    n_show = min(len(cum), max(K + 3, 6))
    a.plot(np.arange(1, n_show + 1), 100 * cum[:n_show], "o-", color=INK, lw=1.5, ms=6)
    a.axhline(100 * args.variance, color=MUTED, ls="--", lw=1)
    a.text(1, 100 * args.variance, f"threshold {100 * args.variance:g}%", va="bottom", ha="left", fontsize=9,
           color=MUTED)
    a.plot(K, 100 * cum[K - 1], "o", ms=11, mfc="none", mec=COLORS[1], mew=2)
    a.annotate(f"K = {K}", (K, 100 * cum[K - 1]), textcoords="offset points", xytext=(10, -18), fontsize=10)
    a.set(xlabel="number of components", ylabel="cumulative explained variance [%]",
          title=f"{lab['b']} choice of the number of components")
    a.set_xticks(np.arange(1, n_show + 1))

    a = AX["c"]
    for k in range(K):
        a.plot(t, W[k], color=COLORS[k % 5], ls=STYLES[k % 5], lw=2,
               label=rf"$\phi_{k + 1}(t)$: {100 * red.explained_variance_ratio[k]:.2f}%")
    a.axhline(0, color=MUTED, lw=0.8)
    a.set(xlabel=tl, ylabel="component (unit norm)", title=f"{lab['c']} principal components")
    a.legend(frameon=False, fontsize=9)

    a = AX.get("d")
    if a is None:
        pass
    elif K >= 2:
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "blues_visible", plt.get_cmap("Blues")(np.linspace(0.3, 1.0, 256)))
        sc = a.scatter(Z[:, 0], Z[:, 1], c=cval, cmap=cmap, s=16, edgecolors="none")
        a.plot(Z[run, 0], Z[run, 1], "o", ms=11, mfc="none", mec=COLORS[0], mew=2)
        a.set(xlabel=r"score $z_1$", ylabel=r"score $z_2$")
        fig.colorbar(sc, ax=a, label=par)
    else:
        a.scatter(cval, Z[:, 0], s=14, color=COLORS[0])
        a.set(xlabel=par, ylabel=r"score $z_1$")
    if a is not None:
        a.set_title(f"{lab['d']} scores of the runs (one point per run, ringed: highlighted)")

    a = AX.get("e")
    for k in range(K if a is not None else 0):
        a.hist(Z[:, k], bins=30, histtype="step", lw=1.8, color=COLORS[k % 5], ls=STYLES[k % 5],
               label=rf"$z_{k + 1}$ (std {Z[:, k].std():.3g})")
    if a is not None:
        a.set(xlabel="score", ylabel="number of runs", title=f"{lab['e']} distribution of the scores")
        a.legend(frameon=False, fontsize=9)

    a = AX["f"]
    a.plot(t, Y[run], color=INK, lw=2.5, label="original run")
    rec = mu.copy()
    a.plot(t, rec, color=MUTED, lw=1.2, ls=":", label="mean only (0 components)")
    for k in range(K):
        rec = rec + Z[run, k] * W[k]
        err = np.sqrt(np.mean((rec - Y[run]) ** 2))
        a.plot(t, rec, color=COLORS[k % 5], ls=STYLES[k % 5], lw=1.8,
               label=f"{k + 1} component{'s' if k else ''} (RMSE {err:.3g})")
    a.set(xlabel=tl, ylabel=args.qoi, title=lab["f"] + r" reconstruction $y = \bar{y} + \sum_k z_k \phi_k$")
    a.legend(frameon=False, fontsize=9)

    for a in AX.values():
        a.grid(alpha=0.25, lw=0.6)
    fig.suptitle(f"PCA of the time-dependent output of {cid}: K = {K} components retained "
                 f"({100 * cum[K - 1]:.2f}% of the variance)", fontsize=13)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=200, bbox_inches="tight")
    print(f"{cid}: K = {K}, explained variance {cum[K - 1]:.4f}; written {args.out}.png/.pdf")
    print("components per case:", dict(sorted(ncomp.items())))


if __name__ == "__main__":
    main()

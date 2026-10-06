"""Step 07 - posterior summaries and plots.

postprocessing.mode: standard | hierarchical | both (latest runs, or
--sc / --hc to choose files). If `synthetic: true`, the true values of
synthetic_truth.h5 are drawn on the plots.

Outputs in results/postprocessing/: posterior_summary.csv, posterior
comparison per parameter, trace plots, predictive plots per case.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from abcf.config import case_noise, output_dir, resolve_path  # noqa: E402
from abcf.io import load_results, load_synthetic_truth  # noqa: E402
from abcf.postprocessing.posterior import Posterior  # noqa: E402
from abcf.workflow import latest_result, load_all, parse_args  # noqa: E402

C_SC, C_HC, C_TRUE, C_EXP = "#e69f00", "#009e73", "#0072b2", "#0b2c5a"


def comparison_plots(posts, truth, setup, figs):
    for j, name in enumerate(setup.calibrated_names):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        if "SC" in posts:
            ax.hist(posts["SC"].theta[:, j], bins=60, density=True, histtype="step", lw=2,
                    color=C_SC, label="standard: posterior")
        if "HC" in posts:
            ax.hist(posts["HC"].population(20000)[:, j], bins=60, density=True, histtype="step", lw=2,
                    color=C_HC, label="hierarchical: population predictive")
            ax.hist(posts["HC"].mu[:, j], bins=60, density=True, alpha=0.25, color=C_HC,
                    label=r"hierarchical: $\mu$")
        if truth is not None:
            k = setup.calibrated_idx[j]
            if truth["population"] is not None:
                ax.axvline(truth["population"][k], color=C_TRUE, lw=2, label="true population value")
            vals = [t[k] for t in truth["cases"].values()]
            if vals:
                ax.plot(vals, np.zeros(len(vals)), "|", ms=18, mew=2, color=C_TRUE,
                        label="true case values")
        ax.set_xlabel(name)
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figs / f"posterior_{name}.png", dpi=150)
        plt.close(fig)


def trace_plots(res, post, figs, stem):
    ch = res["chains"]
    blocks = [("theta", ch["theta"])] if post.method == "SC" else [("mu", ch["mu"]), ("sigma", ch["sigma"])]
    for block, arr in blocks:
        p = arr.shape[2]
        fig, axes = plt.subplots(p, 1, figsize=(10, 2.4 * p), squeeze=False, sharex=True)
        for j in range(p):
            for c in range(min(arr.shape[0], 8)):
                axes[j, 0].plot(arr[c, :, j], lw=0.5, alpha=0.8)
            axes[j, 0].axvline(post.burnin, color="gray", ls="--")
            axes[j, 0].set_ylabel(f"{block}[{post.names[j]}]\n(standardised)", fontsize=8)
        axes[-1, 0].set_xlabel("iteration")
        fig.tight_layout()
        fig.savefig(figs / f"trace_{stem}_{block}.png", dpi=120)
        plt.close(fig)


def predictive_plots(posts, cfg, data, setup, sset, figs, n):
    seed = int(cfg["project"]["seed"])
    cases = next(iter(posts.values())).case_ids
    scalar = all(data.cases[c].is_scalar for c in cases)
    preds = {}
    for m, post in posts.items():
        preds[m] = {}
        for e, cid in enumerate(cases):
            theta = post.case_draws(e, n, seed)
            Xs = sset.scaler.transform(setup.full_vector(theta))
            preds[m][cid] = sset[cid].predict_qoi(Xs)          # (n, n_t)
    rows = []
    if scalar:
        fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(cases) + 3), 4.5))
        x = np.arange(len(cases))
        meas = np.array([data.cases[c].meas[0] for c in cases])
        err = np.array([1.96 * case_noise(cfg, c) * abs(data.cases[c].meas[0]) for c in cases])
        ax.errorbar(x - 0.2, meas, yerr=err, fmt="D", color=C_EXP, capsize=4, label=r"measured $\pm$1.96$\sigma$")
        for dx, (m, col) in zip((0.0, 0.2), (("SC", C_SC), ("HC", C_HC))):
            if m not in preds:
                continue
            q = np.array([np.percentile(preds[m][c][:, 0], [2.5, 50, 97.5]) for c in cases])
            ax.errorbar(x + dx, q[:, 1], yerr=[q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]], fmt="o",
                        color=col, capsize=4, label=f"{m}: 95% interval")
            rows += [{"method": m, "case": c, "q2.5": a, "median": b, "q97.5": d}
                     for c, (a, b, d) in zip(cases, q)]
        ax.set_xticks(x, cases, rotation=45, ha="right")
        ax.set_ylabel("QoI")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figs / "predictive_calibration_cases.png", dpi=150)
        plt.close(fig)
    else:
        for cid in cases:
            case = data.cases[cid]
            fig, ax = plt.subplots(figsize=(9, 4.5))
            ax.errorbar(case.tim, case.meas, yerr=1.96 * case_noise(cfg, cid) * np.abs(case.meas),
                        fmt=".", color=C_EXP, ms=4, lw=0.6, label=r"measured $\pm$1.96$\sigma$")
            for m, col in (("SC", C_SC), ("HC", C_HC)):
                if m in preds:
                    q = np.percentile(preds[m][cid], [2.5, 50, 97.5], axis=0)
                    ax.fill_between(case.tim, q[0], q[2], color=col, alpha=0.3, label=f"{m}: 95% band")
                    ax.plot(case.tim, q[1], color=col, lw=1.2)
            ax.set_xlabel("time")
            ax.set_ylabel("QoI")
            ax.set_title(cid)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(figs / f"predictive_{cid}.png", dpi=150)
            plt.close(fig)
    return pd.DataFrame(rows)


def main():
    args = parse_args(__doc__, lambda p: (p.add_argument("--sc", default=None),
                                          p.add_argument("--hc", default=None)))
    cfg, data, setup, sset = load_all(args.config)
    mode = cfg["postprocessing"]["mode"]
    files = {}
    if mode in ("standard", "both"):
        files["SC"] = Path(args.sc) if args.sc else latest_result(cfg, "SC")
    if mode in ("hierarchical", "both"):
        files["HC"] = Path(args.hc) if args.hc else latest_result(cfg, "HC")
    missing = [m for m, f in files.items() if f is None]
    if missing:
        raise FileNotFoundError(f"No results for {missing}: run step 04/05 or change postprocessing.mode.")

    out = output_dir(cfg, "postprocessing")
    figs = output_dir(cfg, "postprocessing", "figs")
    results = {m: load_results(f) for m, f in files.items()}
    posts = {m: Posterior(r) for m, r in results.items()}

    truth = None
    if cfg["synthetic"]:
        truth = load_synthetic_truth(resolve_path(cfg, cfg["synthetic_truth_file"]), len(setup.names))

    pd.concat([p.summary() for p in posts.values()]).to_csv(out / "posterior_summary.csv", index=False)
    comparison_plots(posts, truth, setup, figs)
    for m, r in results.items():
        trace_plots(r, posts[m], figs, files[m].stem)
    pred = predictive_plots(posts, cfg, data, setup, sset, figs, cfg["postprocessing"]["n_posterior_samples"])
    if len(pred):
        pred.to_csv(out / "predictive_calibration_cases.csv", index=False)
    print(f"Post-processing of {', '.join(f'{m}: {f.name}' for m, f in files.items())}\nResults in {out}")


if __name__ == "__main__":
    main()

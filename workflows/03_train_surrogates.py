"""Step 03 - train and test the GP surrogates of all cases.

Output: results/surrogates/surrogates.pkl (or surrogate.file),
surrogate_summary.csv and, per case, a predicted-vs-simulated plot on the
test runs (not used for training).

    python workflows/03_train_surrogates.py config.yaml --compare-kernels

compares the kernels of ``surrogate.kernel_candidates`` on the same
train/test split (relative RMSE, Q2, standardised residuals, 95% coverage,
NLPD) and writes results/surrogates/kernel_comparison_*.csv and figures;
the GPs are not saved. Set the chosen kernel in ``surrogate.kernel``.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from abcf.config import load_config, output_dir, resolve_path  # noqa: E402
from abcf.io import load_input_data  # noqa: E402
from abcf.surrogate import save_surrogates, train_surrogates  # noqa: E402
from abcf.surrogate.training import compare_kernels  # noqa: E402
from abcf.workflow import parse_args, surrogate_path  # noqa: E402


def kernel_comparison(cfg, data, out, figs):
    from scipy.stats import norm
    print(f"Kernel comparison on {cfg['surrogate'].get('kernel_comparison_cases', 'all')} case(s)")
    per_case, summary, arrays = compare_kernels(data, cfg)
    per_case.to_csv(out / "kernel_comparison_per_case.csv", index=False)
    summary.to_csv(out / "kernel_comparison_summary.csv", index=False)
    print("\n" + summary.to_string(index=False, float_format="%.4g"))
    print(f"\nRecommended (lowest NLPD with standardised-residual std in [0.8, 1.25]): "
          f"{summary.loc[summary['recommended'], 'kernel'].iloc[0]}")
    names = list(arrays)
    k = len(names)
    # parity plots
    fig, axes = plt.subplots(1, k, figsize=(3.6 * k, 3.8), squeeze=False)
    lo = min(a["y"].min() for a in arrays.values())
    hi = max(a["y"].max() for a in arrays.values())
    for ax, n in zip(axes[0], names):
        a = arrays[n]
        ax.plot(a["m"], a["y"], ".", ms=2, alpha=0.5)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.plot([lo, hi], [1.01 * lo, 1.01 * hi], ":", color="0.4", lw=1)
        ax.plot([lo, hi], [0.99 * lo, 0.99 * hi], ":", color="0.4", lw=1)
        ax.set_title(n, fontsize=9)
        ax.set_xlabel("GP prediction")
    axes[0, 0].set_ylabel("simulated (test runs); dotted: +-1%")
    fig.tight_layout()
    fig.savefig(figs / "kernel_comparison_parity.png", dpi=150)
    plt.close(fig)
    # standardised residuals
    fig, axes = plt.subplots(1, k, figsize=(3.6 * k, 3.4), squeeze=False, sharey=True)
    xx = np.linspace(-5, 5, 300)
    for ax, n in zip(axes[0], names):
        z = arrays[n]["z"]
        ax.hist(np.clip(z, -5, 5), bins=60, range=(-5, 5), density=True, alpha=0.6)
        ax.plot(xx, norm.pdf(xx), "r--", lw=1.5, label="N(0,1)")
        ax.set_title(f"{n}\nmean {z.mean():.2f}, std {z.std():.2f}", fontsize=9)
        ax.set_xlabel(r"$(y - m)/\sigma_{GP}$")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "kernel_comparison_residuals.png", dpi=150)
    plt.close(fig)
    # relative RMSE per case
    fig, ax = plt.subplots(figsize=(1.6 * k + 2, 4))
    ax.boxplot([per_case.loc[per_case["kernel"] == n, "rel_rmse_pct"] for n in names])
    ax.set_xticks(range(1, k + 1), names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("relative RMSE on test runs per case [%]")
    fig.tight_layout()
    fig.savefig(figs / "kernel_comparison_rel_rmse.png", dpi=150)
    plt.close(fig)
    print(f"\nResults in {out}")


def main():
    args = parse_args(__doc__, lambda p: p.add_argument(
        "--compare-kernels", action="store_true", help="compare surrogate.kernel_candidates only"))
    cfg = load_config(args.config)
    data = load_input_data(resolve_path(cfg, cfg["data"]["input_file"]))
    out = output_dir(cfg, "surrogates")
    figs = output_dir(cfg, "surrogates", "figs")
    if args.compare_kernels:
        kernel_comparison(cfg, data, out, figs)
        return

    print(f"Training surrogates for {len(data.cases)} case(s), kernel = "
          f"{cfg['surrogate']['kernel']['combine']} of {cfg['surrogate']['kernel']['terms']}")
    sset, summary = train_surrogates(data, cfg)
    save_surrogates(sset, surrogate_path(cfg))
    summary.to_csv(out / "surrogate_summary.csv", index=False)
    if summary["hit_bound"].any():
        print("\nNote: some GPs reached a hyper-parameter bound (refitted with relaxed bounds "
              "if refit_on_bound is true); see surrogate_summary.csv.")

    seed = int(cfg["project"]["seed"])
    for cid, case in data.cases.items():
        X = data.X(case)
        _, idx_test = train_test_split(np.arange(X.shape[0]), test_size=cfg["surrogate"]["test_fraction"],
                                       random_state=seed)
        sur = sset[cid]
        Xs = sset.scaler.transform(X[idx_test])
        z_true = sur.reduction.project(case.pred[idx_test])
        z_pred, z_var = sur.predict(Xs)
        k = z_true.shape[1]
        fig, axes = plt.subplots(1, k, figsize=(4.2 * k, 4), squeeze=False)
        for j in range(k):
            ax = axes[0, j]
            ax.errorbar(z_true[:, j], z_pred[:, j], yerr=1.96 * np.sqrt(z_var[:, j]), fmt="o", ms=4,
                        alpha=0.7, capsize=2)
            lims = [min(z_true[:, j].min(), z_pred[:, j].min()), max(z_true[:, j].max(), z_pred[:, j].max())]
            ax.plot(lims, lims, "k--", lw=1)
            qoi = (cfg.get("labels") or {}).get("QoI", "QoI")
            ax.set_xlabel(f"simulated {qoi}" if case.is_scalar else f"simulated PC{j + 1}")
            ax.set_ylabel(r"GP prediction ($\pm$95% interval)")
        fig.suptitle(f"{cid}: GP on the test runs, " + r"$R^2$" + f" = {sur.test_metrics['r2']:.3f}")
        fig.tight_layout()
        fig.savefig(figs / f"gp_test_{cid}.png", dpi=150)
        plt.close(fig)
    print(f"\nResults in {out}")


if __name__ == "__main__":
    main()

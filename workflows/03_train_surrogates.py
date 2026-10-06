"""Step 03 - train and test the GP surrogates of all cases.

Output: results/surrogates/surrogates.pkl, surrogate_summary.csv and,
per case, a predicted-vs-simulated plot on the held-out runs.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from abcf.config import load_config, output_dir, resolve_path  # noqa: E402
from abcf.io import load_input_data  # noqa: E402
from abcf.surrogate import save_surrogates, train_surrogates  # noqa: E402
from abcf.workflow import parse_args  # noqa: E402


def main():
    args = parse_args(__doc__)
    cfg = load_config(args.config)
    data = load_input_data(resolve_path(cfg, cfg["data"]["input_file"]))
    out = output_dir(cfg, "surrogates")
    figs = output_dir(cfg, "surrogates", "figs")

    print(f"Training surrogates for {len(data.cases)} case(s), kernel = "
          f"{cfg['surrogate']['kernel']['combine']} of {cfg['surrogate']['kernel']['terms']}")
    sset, summary = train_surrogates(data, cfg)
    save_surrogates(sset, out / "surrogates.pkl")
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
            ax.set_xlabel("simulated" if case.is_scalar else f"simulated PC{j + 1}")
            ax.set_ylabel("GP prediction (95%)")
        fig.suptitle(f"{cid}: GP on held-out runs, R2 = {sur.test_metrics['r2']:.3f}")
        fig.tight_layout()
        fig.savefig(figs / f"gp_test_{cid}.png", dpi=150)
        plt.close(fig)
    print(f"\nResults in {out}")


if __name__ == "__main__":
    main()

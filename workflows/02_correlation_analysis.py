"""Step 02 - correlation analysis on the training samples.

1. input-output: Spearman correlation (BH-FDR) between parameters and the
   outputs of every case (QoI, or PCA scores for a time-dependent QoI);
2. output-output: Spearman correlation between the outputs of every pair
   of cases, redundancy groups and a suggested calibration/validation split.

Outputs in results/correlation/. Nothing is decided automatically: the
tables support the choice of the calibrated parameters (parameters block)
and of the cases (data block) in the configuration.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from abcf.config import load_config, output_dir, resolve_path  # noqa: E402
from abcf.io import load_input_data  # noqa: E402
from abcf.sensitivity import correlation as corr  # noqa: E402
from abcf.workflow import parse_args  # noqa: E402


def heatmap(table, case_id, path):
    sub = table[table["case"] == case_id]
    piv = sub.pivot(index="parameter", columns="output", values="spearman_rho")
    sig = sub.pivot(index="parameter", columns="output", values="influential")
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * piv.shape[1], 0.6 * piv.shape[0] + 1.5))
    im = ax.imshow(piv.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            txt = f"{piv.values[i, j]:.2f}" + ("" if sig.values[i, j] else "\nn.i.")
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)
    ax.set_xticks(range(piv.shape[1]), piv.columns)
    ax.set_yticks(range(piv.shape[0]), piv.index)
    ax.set_title(f"{case_id}: Spearman rho (n.i. = not influential)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args(__doc__)
    cfg = load_config(args.config)
    c = cfg["sensitivity"]["correlation"]
    data = load_input_data(resolve_path(cfg, cfg["data"]["input_file"]))
    out = output_dir(cfg, "correlation")
    figs = output_dir(cfg, "correlation", "figs")

    outputs = corr.case_outputs(data, cfg["surrogate"]["pca_variance"])

    io_table = corr.input_output(data, outputs, c)
    io_table.to_csv(out / "input_output_per_case.csv", index=False)
    summary = corr.input_output_summary(io_table)
    summary.to_csv(out / "input_output_summary.csv")
    for cid in data.case_ids:
        heatmap(io_table, cid, figs / f"input_output_{cid}.png")
    print("\nInput-output summary across cases:")
    print(summary.to_string(float_format="%.3f"))
    always = summary.index[summary["influential_in_all_cases"]].tolist()
    print(f"\nInfluential in ALL cases: {always}")

    if len(outputs) > 1:
        oo = corr.output_output(outputs, c)
        oo.to_csv(out / "output_output_pairs.csv", index=False)
        groups = corr.redundancy_groups(oo, data.case_ids, outputs)
        groups.to_csv(out / "redundancy_groups.csv", index=False)
        n_red = int(oo["redundant"].sum())
        print(f"\nOutput-output: {n_red}/{len(oo)} redundant pairs "
              f"(|rho| >= {c['redundancy_threshold']}, FDR {c['alpha']}).")
        print(f"Suggested split: {int(groups['representative'].sum())} calibration, "
              f"{int((~groups['representative']).sum())} validation (see redundancy_groups.csv).")

        first = oo[oo["output_index"] == 0]
        ids = data.case_ids
        mat = np.eye(len(ids))
        pos = {k: i for i, k in enumerate(ids)}
        for _, r in first.iterrows():
            mat[pos[r["case_A"]], pos[r["case_B"]]] = mat[pos[r["case_B"]], pos[r["case_A"]]] = r["abs_rho"]
        fig, ax = plt.subplots(figsize=(0.35 * len(ids) + 3, 0.35 * len(ids) + 2.5))
        im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(ids)), ids, rotation=90, fontsize=7)
        ax.set_yticks(range(len(ids)), ids, fontsize=7)
        ax.set_title("|Spearman rho| between cases (first output)")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(figs / "output_output.png", dpi=150)
        plt.close(fig)
    print(f"\nResults in {out}")


if __name__ == "__main__":
    main()

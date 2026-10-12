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

from abcf.config import load_config, output_dir, param_label, resolve_path  # noqa: E402
from abcf.io import load_input_data  # noqa: E402
from abcf.sensitivity import correlation as corr  # noqa: E402
from abcf.workflow import parse_args  # noqa: E402


def _natural(cid):
    """Sort key: ROD2 before ROD10."""
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(cid))]


def heatmap(table, case_id, path, cfg):
    sub = table[table["case"] == case_id]
    order = list(dict.fromkeys(sub["parameter"]))          # order of the data file
    piv = sub.pivot(index="parameter", columns="output", values="rho_squared").loc[order]
    sgn = sub.pivot(index="parameter", columns="output", values="spearman_rho").loc[order]
    sig = sub.pivot(index="parameter", columns="output", values="influential").loc[order]
    fig, ax = plt.subplots(figsize=(3.2 + 1.1 * piv.shape[1], 0.6 * piv.shape[0] + 1.6))
    im = ax.imshow(piv.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            txt = f"{v:.3f} ({'+' if sgn.values[i, j] >= 0 else '-'})" + ("" if sig.values[i, j] else "\nn.i.")
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="white" if v > 0.6 else "black")
    ax.set_xticks(range(piv.shape[1]), piv.columns)
    ax.set_yticks(range(piv.shape[0]), [param_label(cfg, n) for n in piv.index])
    ax.set_title(f"{case_id}: " + r"$\rho_S^2$" + " (sign of " + r"$\rho_S$" + ")\n(n.i. = not influential)",
                 fontsize=10)
    fig.colorbar(im, ax=ax, label=r"$\rho_S^2$")
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
        heatmap(io_table, cid, figs / f"input_output_{cid}.png", cfg)
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
              f"(rho^2 >= {c['redundancy_threshold']}, FDR {c['alpha']}).")
        print(f"Suggested split: {int(groups['representative'].sum())} calibration, "
              f"{int((~groups['representative']).sum())} validation (see redundancy_groups.csv).")
        n_val = int(c.get("n_validation") or 0)
        if n_val > 0:
            split = corr.validation_split(oo, data.case_ids, n_val)
            split.to_csv(out / "validation_split.csv", index=False)
            val = split[split["set"] == "validation"]
            print(f"\nValidation split (lowest maximum significant rho^2): {n_val} validation cases")
            print(val.to_string(index=False, float_format="%.5f"))
            print("Use it with  data: {validation_cases: <path to validation_split.csv>}")

        first = oo[oo["output_index"] == 0]
        ids = sorted(data.case_ids, key=_natural)
        mat = np.full((len(ids), len(ids)), np.nan)
        pos = {k: i for i, k in enumerate(ids)}
        for _, r in first.iterrows():
            i, j = sorted((pos[r["case_A"]], pos[r["case_B"]]))
            mat[j, i] = r["rho_squared"]                      # lower triangle only
        vmin = c.get("heatmap_vmin", "auto")
        if vmin in (None, "auto"):
            vmin = np.floor(np.nanmin(mat) * 100) / 100
        n = len(ids)
        fig, ax = plt.subplots(figsize=(min(0.3 * n + 3, 30), min(0.3 * n + 2.5, 28)))
        im = ax.imshow(np.ma.masked_invalid(mat), cmap="viridis", vmin=float(vmin), vmax=1)
        step = max(1, n // 40)                            # at most ~40 tick labels per axis
        ax.set_xticks(range(0, n, step), ids[::step], rotation=90, fontsize=7)
        ax.set_yticks(range(0, n, step), ids[::step], fontsize=7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(r"$\rho_S^2$ between the outputs of the cases (first output)")
        fig.colorbar(im, ax=ax, shrink=0.8, label=r"$\rho_S^2$")
        fig.tight_layout()
        fig.savefig(figs / "output_output.png", dpi=200)
        fig.savefig(figs / "output_output.pdf")
        plt.close(fig)
    print(f"\nResults in {out}")


if __name__ == "__main__":
    main()

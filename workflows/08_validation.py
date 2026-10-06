"""Step 08 - predictive validation with solver re-runs.

The user re-runs the solver with posterior samples (export them with
--export-samples, then fill data/validation_SC.h5 and validation_HC.h5,
format in docs/DATA_FORMAT.md). This script computes, per case and
method, z-scores, coverage of the predictive interval and RMSE, and
compares SC and HC on their common cases.

  python workflows/08_validation.py config.yaml --export-samples 200
  python workflows/08_validation.py config.yaml
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from abcf.config import case_noise, load_config, output_dir, parameter_setup, resolve_path  # noqa: E402
from abcf.io import load_input_data, load_results, load_validation  # noqa: E402
from abcf.postprocessing.posterior import Posterior  # noqa: E402
from abcf.validation.metrics import case_metrics, validation_table  # noqa: E402
from abcf.workflow import latest_result, parse_args  # noqa: E402


def export(cfg, n):
    """Posterior draws for a NEW case (SC: theta; HC: population), full parameter vectors."""
    data = load_input_data(resolve_path(cfg, cfg["data"]["input_file"]))
    setup = parameter_setup(cfg, data.param_names)
    out = output_dir(cfg, "validation")
    for m in ("SC", "HC"):
        f = latest_result(cfg, m)
        if f is None:
            continue
        draws = Posterior(load_results(f)).draws(n, int(cfg["project"]["seed"]))
        df = pd.DataFrame(setup.full_vector(draws), columns=setup.names)
        df.insert(0, "sample_id", np.arange(n))
        df.to_csv(out / f"posterior_samples_{m}.csv", index=False)
        print(f"{m}: {n} samples -> {out / f'posterior_samples_{m}.csv'} (run the solver with them)")


def main():
    args = parse_args(__doc__, lambda p: p.add_argument("--export-samples", type=int, default=None))
    cfg = load_config(args.config)
    if args.export_samples:
        export(cfg, args.export_samples)
        return
    v = cfg["validation"]
    if not v["run"]:
        print("validation.run is false in the config: nothing to do.")
        return
    out = output_dir(cfg, "validation")
    figs = output_dir(cfg, "validation", "figs")
    results = {}
    for method, file in (v["files"] or {}).items():
        path = resolve_path(cfg, file)
        if not path.exists():
            print(f"{method}: {path} not found, skipped.")
            continue
        results[method] = {cid: case_metrics(c["pred"], c["meas"], case_noise(cfg, cid),
                                             v["interval_level"], int(cfg["project"]["seed"]))
                           for cid, c in load_validation(path).items()}
        for cid, c in load_validation(path).items():
            m = results[method][cid]
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.axhspan(-1.96, 1.96, color="gray", alpha=0.15)
            ax.plot(c["tim"] if len(c["tim"]) == len(m["z"]) else np.arange(len(m["z"])), m["z"], "o-", ms=3)
            ax.set_ylabel("z-score")
            ax.set_title(f"{method} - {cid}: coverage {m['coverage']:.2f}, RMSE {m['rmse']:.3g}")
            fig.tight_layout()
            fig.savefig(figs / f"zscore_{method}_{cid}.png", dpi=130)
            plt.close(fig)
    if not results:
        raise FileNotFoundError("No validation file found (validation.files in the config).")
    table = validation_table(results)
    table.to_csv(out / "validation_metrics.csv", index=False)
    print(table.to_string(index=False, float_format="%.4g"))
    if {"SC", "HC"} <= set(results):
        common = sorted(set(results["SC"]) & set(results["HC"]))
        comp = table[table["case"].isin(common)].groupby("method")[["coverage", "rmse", "mean_abs_z"]].mean()
        comp.to_csv(out / "validation_SC_vs_HC.csv")
        print(f"\nSC vs HC on {len(common)} common case(s):\n{comp.to_string(float_format='%.4g')}")


if __name__ == "__main__":
    main()

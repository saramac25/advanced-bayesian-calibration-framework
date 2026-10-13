"""Step 08 - predictive validation.

validation.source = solver: the user re-runs the solver with posterior
samples (export them with --export-samples, then fill
data/validation_SC.h5 and validation_HC.h5, format in
docs/DATA_FORMAT.md). This script computes, per case and method,
z-scores, coverage of the predictive interval and RMSE, and compares SC
and HC on their common cases.

validation.source = surrogate: the posterior (SC: posterior samples; HC:
new-case draws theta ~ N(mu, sigma) truncated to the prior range, mu and
sigma from the same iteration) is propagated through the GPs of the
validation cases (data.validation_cases), `validation.n_draws` draws.
Output: chi2_E and chi2_tot (MAP prediction, sigma_fit^2 = sigma_post^2 +
sigma_GP^2), coverage, z-scores, RMSE, per case and per method.

  python workflows/08_validation.py config.yaml --export-samples 93   (Wilks 95/95 two-sided)
  python workflows/08_validation.py config.yaml
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from abcf.config import (case_noise, load_config, output_dir, parameter_setup, resolve_path,  # noqa: E402
                         validation_case_list)
from abcf.io import load_input_data, load_results, load_validation  # noqa: E402
from abcf.postprocessing.posterior import Posterior  # noqa: E402
from abcf.validation.metrics import case_metrics, chi2_terms, validation_table  # noqa: E402
from abcf.workflow import latest_result, load_all, parse_args  # noqa: E402


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


def surrogate_validation(config_path):
    """Validation through the GPs of the validation cases (chi2_E, chi2_tot, coverage)."""
    cfg, data, setup, sset = load_all(config_path)
    v = cfg["validation"]
    seed = int(cfg["project"]["seed"])
    val_cases = validation_case_list(cfg)
    if not val_cases:
        raise ValueError("validation.source = surrogate needs data.validation_cases.")
    out = output_dir(cfg, "validation")
    rows, summ = [], []
    for m in ("SC", "HC"):
        f = latest_result(cfg, m)
        if f is None:
            continue
        post = Posterior(load_results(f))
        draws = post.draws(int(v["n_draws"]), seed)
        theta_map = post.map_point()
        for cid in val_cases:
            case = data.cases[cid]
            sur = sset[cid]
            y = case.meas
            mean_d, var_d = sur.predict(sset.scaler.transform(setup.full_vector(draws, cid)))
            z_map, var_map = sur.predict(sset.scaler.transform(setup.full_vector(theta_map, cid)))
            red = sur.reduction
            pred_d = red.reconstruct(mean_d)                         # (n_draws, n_t)
            p_fit = red.reconstruct(z_map)[0]
            # GP std in QoI space: variance of the components mapped through W (diagonal)
            sig_gp = np.sqrt(np.maximum((var_map @ red.W ** 2)[0], 0.0))
            sig_post = pred_d.std(axis=0, ddof=1)
            sig_exp = case_noise(cfg, cid) * np.abs(y)
            t = chi2_terms(y, p_fit, sig_post, sig_gp, sig_exp)
            # predictive samples including the GP variance, for coverage and z-scores
            rng = np.random.default_rng(seed)
            noisy_gp = pred_d + rng.standard_normal(pred_d.shape) * np.sqrt(
                np.maximum(var_d @ red.W ** 2, 0.0))
            cm = case_metrics(noisy_gp, y, case_noise(cfg, cid), v["interval_level"], seed)
            for k in range(len(y)):
                rows.append({"method": m, "case": cid, "point": k, "y_meas": y[k], "p_fit_MAP": p_fit[k],
                             "sigma_post": sig_post[k], "sigma_GP": sig_gp[k], "sigma_exp": sig_exp[k],
                             "sigma_fit": t["sigma_fit"][k],
                             "chi2_E_term": t["chi2_E_term"][k], "chi2_tot_term": t["chi2_tot_term"][k],
                             "pred_mean": cm["mean"][k], "pred_q_lo": cm["q_lo"][k], "pred_q_hi": cm["q_hi"][k],
                             "inside_interval": bool(cm["inside"][k]), "z": cm["z"][k]})
        d = pd.DataFrame([r for r in rows if r["method"] == m])
        summ.append({"method": m, "file": f.name, "n_cases": len(val_cases), "n_points": len(d),
                     "chi2_E": d["chi2_E_term"].mean(), "chi2_tot": d["chi2_tot_term"].mean(),
                     "sqrt_chi2_E": np.sqrt(d["chi2_E_term"].mean()),
                     "coverage": d["inside_interval"].mean(), "mean_abs_z": d["z"].abs().mean(),
                     "rmse": float(np.sqrt(np.mean((d["pred_mean"] - d["y_meas"]) ** 2))),
                     "MAP": "; ".join(f"{n}={x:.5g}" for n, x in zip(post.names, theta_map))})
    if not summ:
        raise FileNotFoundError("No calibration results found: run step 04 and/or 05 first.")
    pd.DataFrame(rows).to_csv(out / "validation_surrogate_per_point.csv", index=False)
    table = pd.DataFrame(summ)
    table.to_csv(out / "validation_surrogate_summary.csv", index=False)
    print(f"Validation through the GPs of {len(val_cases)} case(s), {v['n_draws']} draws "
          f"(ideal chi2 = 1):")
    print(table.drop(columns=["file"]).to_string(index=False, float_format="%.4g"))
    print(f"\nResults in {out}")


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
    if v.get("source", "solver") == "surrogate":
        surrogate_validation(args.config)
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

"""Step 05 - hierarchical Bayesian calibration (MH-within-Gibbs).

Case-level parameters X[e] ~ N(mu, sigma^2), population mean mu and
standard deviation sigma inferred jointly. Several independent chains
(different seeds, dispersed starting points if requested).

With ``hierarchical.nsteps_list`` the lengths are tried in order until
the convergence criteria of the ``diagnostics`` block are met (R-hat,
bulk/tail ESS, N >= tau_factor * tau on the required blocks); only the
final run is saved, the table of all lengths goes to
HC_<timestamp>_lengths.csv. The acceptance is checked at the final
length (step 06 prints suggestions if it is out of range).

Output: results/calibration/HC_<timestamp>.h5
"""

import numpy as np
import pandas as pd

from abcf.calibration.hierarchical.runner import run_hierarchical
from abcf.calibration.likelihood import build_likelihoods
from abcf.diagnostics.criteria import hierarchical_rows
from abcf.config import output_dir
from abcf.io import save_results
from abcf.workflow import calibration_cases, load_all, parse_args, qoi_type, timestamp


def main():
    args = parse_args(__doc__, lambda p: p.add_argument(
        "--no-progress", action="store_true", help="hide the progress bars"))
    cfg, data, setup, sset = load_all(args.config)
    cases = calibration_cases(cfg, data)
    idx = setup.calibrated_idx
    lower, upper = sset.scaler.lower[idx], sset.scaler.upper[idx]
    h = cfg["hierarchical"]

    print(f"Hierarchical calibration of {setup.calibrated_names} on {len(cases)} case(s); "
          f"fixed: {setup.fixed_values or 'none'}"
          + (f" (per-case values from data.fixed_values_file for {len(setup.case_fixed)} case(s))"
             if setup.case_fixed else ""))
    lengths = list(h.get("nsteps_list") or [h["nsteps"]])
    print(f"{h['n_chains']} chains x {lengths} steps, burn-in {h['burnin_fraction']:.0%}, "
          f"adaptation: mu={h['adaptation']['mu']}, sigma={h['adaptation']['sigma']}, X={h['adaptation']['X']}, "
          f"joint shift move: {h['joint_shift_move']['enabled']}, "
          f"joint scale move: {(h.get('joint_scale_move') or {}).get('enabled', False)}")
    from abcf.config import normal_prior_std
    prior = normal_prior_std(cfg, setup.names, sset.scaler, idx)
    if prior is not None:
        h["_normal_prior_mu"] = [prior[0].tolist(), prior[1].tolist()]
        print("Normal priors on mu (physical units): " + ", ".join(
            f"{n} ~ N({cfg['parameters'][n]['prior']['mean']}, {cfg['parameters'][n]['prior']['std']}^2)"
            for n in setup.calibrated_names if (cfg["parameters"][n].get("prior") or {}).get("type") == "normal")
            + ", truncated to the training range")
    liks = build_likelihoods(cfg, data, sset, setup, cases)
    table = []
    for n in lengths:
        res = run_hierarchical(liks, cases, lower, upper, cfg, progress=not args.no_progress, nsteps=n)
        rows = hierarchical_rows(res, res["burnin"], setup.calibrated_names, cases, cfg)
        req = [r for r in rows if r["convergence_required"]]
        conv = all(r["converged"] for r in req)
        acc_ok = all(r["acceptance_ok"] for r in rows)
        worst = min(req, key=lambda r: r["ess_bulk"])
        table.append({"nsteps": n, "converged": conv, "acceptance_ok": acc_ok,
                      "rhat_max": max(r["rhat"] for r in req), "ess_bulk_min": worst["ess_bulk"],
                      "ess_tail_min": min(r["ess_tail"] for r in req),
                      "n_over_tau_min": min(r["n_over_tau"] for r in req),
                      "wall_clock_s_max_chain": float(np.max(res["wall_clock_s"]))})
        print(f"  nsteps = {n}: converged = {conv} (R-hat max {table[-1]['rhat_max']:.4f}, "
              f"ESS min {worst['ess_bulk']:.0f}, N/tau min {table[-1]['n_over_tau_min']:.0f}), "
              f"acceptance in range = {acc_ok}, {table[-1]['wall_clock_s_max_chain'] / 60:.1f} min/chain")
        if conv:
            break
    if not table[-1]["converged"]:
        print("WARNING: the longest length does not meet the convergence criteria (see step 06).")

    stamp = timestamp()
    pd.DataFrame(table).to_csv(output_dir(cfg, "calibration") / f"HC_{stamp}_lengths.csv", index=False)
    path = save_results(output_dir(cfg, "calibration") / f"HC_{stamp}.h5", {
        "metadata": {
            "method": "HC", "qoi_type": qoi_type(data, cases), "case_ids": cases,
            "param_names": setup.calibrated_names, "all_param_names": setup.names,
            "calibrated_idx": np.asarray(idx), "fixed_parameters": str(setup.fixed_values),
            "include_gp_variance": bool(cfg["likelihood"]["include_gp_variance"]),
            "relative_noise": float(cfg["measurement_noise"]["relative"]),
            "n_chains": int(h["n_chains"]), "nsteps": res["nsteps"], "burnin": res["burnin"],
            "adaptation": f"mu={h['adaptation']['mu']}, sigma={h['adaptation']['sigma']}, X={h['adaptation']['X']}", "joint_shift_move": bool(h["joint_shift_move"]["enabled"]),
            "joint_scale_move": bool((h.get("joint_scale_move") or {}).get("enabled", False)),
            "seeds": res["seeds"], "wall_clock_s": res["wall_clock_s"],
        },
        "scaler": {"mean": sset.scaler.mean, "std": sset.scaler.std,
                   "lower": sset.scaler.lower, "upper": sset.scaler.upper},
        "chains": {"mu": res["mu"], "sigma": res["sigma"], "X": res["X"].astype(np.float32),
                   "logpost": res["logpost"],
                   "acceptance_shift": res["acceptance_shift"].astype(np.int8),
                   "acceptance_mu": res["acceptance_mu"].astype(np.int8),
                   "acceptance_sigma": res["acceptance_sigma"].astype(np.int8),
                   "acceptance_X": res["acceptance_X"].astype(np.int8),
                   "shift_std_final": res["shift_std_final"],
                   "acceptance_scale": res["acceptance_scale"].astype(np.int8),
                   "scale_std_final": res["scale_std_final"]},
    })
    print(f"\nWall clock per chain: {np.round(res['wall_clock_s'], 1)} s\nSaved: {path}")


if __name__ == "__main__":
    main()

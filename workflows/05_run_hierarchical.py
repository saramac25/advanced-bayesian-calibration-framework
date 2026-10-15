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

With ``hierarchical.continue_chains: true`` (default) a longer length does not
restart the chains: they continue from the end of the previous length, with the
proposals frozen at the end of the first adaptation (e.g. 10k -> 30k runs 20k
new iterations). The burn-in is always burnin_fraction x total length, so the
adaptation phase (first segment) stays inside the discarded part.

``--continue-from FILE`` (or ``--continue-from latest``) continues, in a new run, the
chains saved in a previous HC_*.h5 (e.g. 50k -> 100k without repeating the first 50k):
only the lengths of nsteps_list longer than the saved chains are run. The file must
come from the same configuration (same cases, parameters, number of chains) and from
a version that saves the chain checkpoints. The previous file is never modified; the
continued chains are saved in a new HC_<timestamp>.h5.

Output: results/calibration/HC_<timestamp>.h5
"""

import numpy as np
import pandas as pd

from abcf.calibration.hierarchical.runner import (checkpoints_for_h5, previous_from_saved,
                                                  run_hierarchical)
from abcf.calibration.likelihood import build_likelihoods
from abcf.diagnostics.criteria import hierarchical_rows
from abcf.config import output_dir
from abcf.io import load_results, save_results
from abcf.workflow import calibration_cases, latest_result, load_all, parse_args, qoi_type, timestamp


def main():
    def extra(p):
        p.add_argument("--no-progress", action="store_true", help="hide the progress bars")
        p.add_argument("--continue-from", default=None, metavar="FILE",
                       help="continue the chains of a previous HC_*.h5 ('latest' = most recent)")
    args = parse_args(__doc__, extra)
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
    cont = bool(h.get("continue_chains", True))
    if cont and len(lengths) > 1:
        print("Longer lengths continue the previous chains (hierarchical.continue_chains).")
    res = None
    if args.continue_from:
        src = latest_result(cfg, "HC") if args.continue_from == "latest" else args.continue_from
        if src is None:
            raise SystemExit("--continue-from latest: no HC_*.h5 in the calibration folder.")
        saved = load_results(src)
        md = saved["metadata"]
        same = (list(md["case_ids"]) == list(cases) and list(md["param_names"]) == setup.calibrated_names
                and int(md["n_chains"]) == int(h["n_chains"])
                and np.allclose(saved["scaler"]["mean"], sset.scaler.mean)
                and np.allclose(saved["scaler"]["std"], sset.scaler.std))
        if not same:
            raise SystemExit(f"{src}: different cases, parameters, number of chains or surrogates: "
                             "it cannot be continued with this configuration.")
        try:
            res = previous_from_saved(saved)
        except ValueError as err:
            raise SystemExit(f"{src}: {err}")
        cont = True
        lengths = [n for n in lengths if n > res["nsteps"]]
        print(f"Continuing the {h['n_chains']} chains of {src} ({res['nsteps']} steps) "
              f"to {lengths} steps.")
        if not lengths:
            raise SystemExit("No length in nsteps_list is longer than the saved chains: nothing to do.")
    for n in lengths:
        res = run_hierarchical(liks, cases, lower, upper, cfg, progress=not args.no_progress, nsteps=n,
                               previous=res if cont else None)
        cont = bool(h.get("continue_chains", True)) or args.continue_from is not None
        rows = hierarchical_rows(res, res["burnin"], setup.calibrated_names, cases, cfg)
        req = [r for r in rows if r["convergence_required"]]
        conv = all(r["converged"] for r in req)
        acc_ok = all(r["acceptance_ok"] for r in rows)
        worst = min(req, key=lambda r: r["ess_bulk"])
        table.append({"nsteps": n, "converged": conv, "acceptance_ok": acc_ok,
                      "rhat_max": max(r["rhat"] for r in req), "ess_bulk_min": worst["ess_bulk"],
                      "ess_tail_min": min(r["ess_tail"] for r in req),
                      "n_over_tau_min": min(r["n_over_tau"] for r in req),
                      "wall_clock_s_max_chain": float(np.max(res["wall_clock_s"])),
                      "continued_from": int(res.get("continued_from", 0))})
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
            "continued_chains": bool(cont), "lengths_run": str([t["nsteps"] for t in table]),
            "continued_from_file": str(src) if args.continue_from else "",
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
        # last state, random-number state and frozen proposals of each chain (--continue-from)
        "checkpoints": checkpoints_for_h5(res["checkpoints"]),
    })
    print(f"\nWall clock per chain: {np.round(res['wall_clock_s'], 1)} s\nSaved: {path}")


if __name__ == "__main__":
    main()

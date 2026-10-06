"""Step 05 - hierarchical Bayesian calibration (MH-within-Gibbs).

Case-level parameters X[e] ~ N(mu, sigma^2), population mean mu and
standard deviation sigma inferred jointly. Several independent chains
(different seeds, dispersed starting points if requested).

Output: results/calibration/HC_<timestamp>.h5
"""

import numpy as np

from abcf.calibration.hierarchical.runner import run_hierarchical
from abcf.calibration.likelihood import build_likelihoods
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
          f"fixed: {setup.fixed_values or 'none'}")
    print(f"{h['n_chains']} chains x {h['nsteps']} steps, burn-in {h['burnin_fraction']:.0%}, "
          f"adaptation: mu={h['adaptation']['mu']}, sigma={h['adaptation']['sigma']}, X={h['adaptation']['X']}, "
          f"joint shift move: {h['joint_shift_move']['enabled']}")
    liks = build_likelihoods(cfg, data, sset, setup, cases)
    res = run_hierarchical(liks, cases, lower, upper, cfg, progress=not args.no_progress)

    stamp = timestamp()
    path = save_results(output_dir(cfg, "calibration") / f"HC_{stamp}.h5", {
        "metadata": {
            "method": "HC", "qoi_type": qoi_type(data, cases), "case_ids": cases,
            "param_names": setup.calibrated_names, "all_param_names": setup.names,
            "calibrated_idx": np.asarray(idx), "fixed_parameters": str(setup.fixed_values),
            "include_gp_variance": bool(cfg["likelihood"]["include_gp_variance"]),
            "relative_noise": float(cfg["measurement_noise"]["relative"]),
            "n_chains": int(h["n_chains"]), "nsteps": res["nsteps"], "burnin": res["burnin"],
            "adaptation": f"mu={h['adaptation']['mu']}, sigma={h['adaptation']['sigma']}, X={h['adaptation']['X']}", "joint_shift_move": bool(h["joint_shift_move"]["enabled"]),
            "seeds": res["seeds"], "wall_clock_s": res["wall_clock_s"],
        },
        "scaler": {"mean": sset.scaler.mean, "std": sset.scaler.std,
                   "lower": sset.scaler.lower, "upper": sset.scaler.upper},
        "chains": {"mu": res["mu"], "sigma": res["sigma"], "X": res["X"], "logpost": res["logpost"],
                   "acceptance_shift": res["acceptance_shift"].astype(np.int8),
                   "acceptance_mu": res["acceptance_mu"].astype(np.int8),
                   "acceptance_sigma": res["acceptance_sigma"].astype(np.int8),
                   "acceptance_X": res["acceptance_X"].astype(np.int8),
                   "shift_std_final": res["shift_std_final"]},
    })
    print(f"\nWall clock per chain: {np.round(res['wall_clock_s'], 1)} s\nSaved: {path}")


if __name__ == "__main__":
    main()

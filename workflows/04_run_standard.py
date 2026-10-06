"""Step 04 - standard Bayesian calibration (emcee).

One set of calibrated parameters for all calibration cases. The number
of walkers and of steps is tuned in two steps (see
abcf/calibration/standard/emcee_runner.py).

Output: results/calibration/SC_<timestamp>.h5 and SC_<timestamp>_tuning.csv
"""

import numpy as np
import pandas as pd

from abcf.calibration.likelihood import build_likelihoods
from abcf.calibration.standard.emcee_runner import StandardPosterior, tune_and_run
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

    print(f"Standard calibration of {setup.calibrated_names} on {len(cases)} case(s); "
          f"fixed: {setup.fixed_values or 'none'}")
    liks = build_likelihoods(cfg, data, sset, setup, cases)
    post = StandardPosterior(liks, lower, upper)
    run, tuning = tune_and_run(post, lower, upper, cfg, setup.n_calibrated, not args.no_progress)

    stamp = timestamp()
    out = output_dir(cfg, "calibration")
    pd.DataFrame(tuning).to_csv(out / f"SC_{stamp}_tuning.csv", index=False)
    path = save_results(out / f"SC_{stamp}.h5", {
        "metadata": {
            "method": "SC", "qoi_type": qoi_type(data, cases), "case_ids": cases,
            "param_names": setup.calibrated_names, "all_param_names": setup.names,
            "calibrated_idx": np.asarray(idx), "fixed_parameters": str(setup.fixed_values),
            "include_gp_variance": bool(cfg["likelihood"]["include_gp_variance"]),
            "relative_noise": float(cfg["measurement_noise"]["relative"]),
            "nwalkers": run["nwalkers"], "nsteps": run["nsteps"], "burnin": run["burnin"],
            "stretch_a": run["stretch_a"], "seed": run["seed"], "wall_clock_s": run["wall_clock_s"],
            "n_chains": run["nwalkers"],
        },
        "scaler": {"mean": sset.scaler.mean, "std": sset.scaler.std,
                   "lower": sset.scaler.lower, "upper": sset.scaler.upper},
        "chains": {"theta": run["chains"], "log_prob": run["log_prob"],
                   "acceptance": run["acceptance"]},
    })
    print(f"\nnwalkers = {run['nwalkers']}, nsteps = {run['nsteps']}, "
          f"wall clock = {run['wall_clock_s']:.1f} s\nSaved: {path}")


if __name__ == "__main__":
    main()

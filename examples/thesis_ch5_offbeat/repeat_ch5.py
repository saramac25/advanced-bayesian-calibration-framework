"""Repeated calibrations of a Chapter 5 case: coverage of the true parameters over N
independent synthetic data sets (frequentist check of the 95% intervals).

Every repetition r = 0 ... N-1 uses the same calibration rods and model (configuration of
the case, e.g. config_modelbias_uniform_k.yaml) and a NEW realisation of the 1% measurement
noise of the synthetic experiments:

    y_i^(r) = FCT_i^model(true vector A) x (1 + 0.01 eps_i^(r)),   eps ~ N(0, 1), seed seed0 + r

(the noiseless OFFBEAT values are FCT_EOL_model_K of inputs/synthetic_experiments.csv).
The standard (emcee) and hierarchical (MH-within-Gibbs) calibrations are run at FIXED lengths,
the ones at which the reference run of the case converged; the convergence is NOT checked
again in the repetitions. For each repetition and calibrated parameter the script stores

    SC  : posterior mean, 95% interval, truth inside?
    HC  : population mean mu (mean, 95% interval, truth inside?), population spread sigma
          (mean), and the 95% interval of the population predictive theta_new = mu + sigma z
          (the distribution used to predict a new rod), truth inside?

and at the end the coverage over the repetitions (results/repeat_<case>/coverage.csv).

    # reference run of each case (convergence length), then the repetitions at that length
    python run_thesis_ch5.py modelbias_uniform_k modelbias_struct_k --n-cal 16 --hc-steps 10000 30000 50000
    python repeat_ch5.py modelbias_uniform_k --n-cal 16 --reps 100 --jobs 14
    python repeat_ch5.py modelbias_struct_k --n-cal 16 --reps 100 --jobs 14
    python figures_bias_ch5.py           # coverage and rod-level figures of the two cases

Each repetition runs its 4 hierarchical chains in parallel (4 cores): --jobs 15 uses 60 cores.
The repetitions already done are skipped, so an interrupted run is resumed by launching the
same command again. --delete-chains removes the SC/HC files of a repetition once summarised
(about 30 MB per repetition otherwise).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_thesis_ch5 import WF, calibration_subset, effective_config  # noqa: E402

from abcf.io import load_results  # noqa: E402

TRUE_A = json.loads((HERE / "inputs" / "metadata.json").read_text())["true_A"]
# one thread per process: the parallelism is over chains and repetitions
ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
           NUMEXPR_NUM_THREADS="1")


def reference_lengths(results: Path, case: str):
    """(hc_steps, sc_steps, sc_walkers) of the converged reference run of the case."""
    cal = results / case / "calibration"
    hl, st = sorted(cal.glob("HC_*_lengths.csv")), sorted(cal.glob("SC_*_tuning.csv"))
    if not hl or not st:
        sys.exit(f"No reference run in {cal}: run run_thesis_ch5.py {case} first, "
                 "or give --hc-steps, --sc-steps and --sc-walkers.")
    h = pd.read_csv(hl[-1])
    conv = h[h["converged"]]
    hc = int((conv if len(conv) else h).iloc[0 if len(conv) else -1]["nsteps"])
    s = pd.read_csv(st[-1])
    s = s[s["step"] == "2_steps"]
    sconv = s[s["converged"]]
    row = (sconv if len(sconv) else s).iloc[0 if len(sconv) else -1]
    if not len(conv) or not len(sconv):
        print("WARNING: the reference run did not converge at any length: the longest is used.")
    return hc, int(row["nsteps"]), int(row["nwalkers"])


def make_data(src: Path, dst: Path, seed: int, noiseless: pd.Series, rel: float):
    shutil.copy(src, dst)
    rng = np.random.default_rng(seed)
    with h5py.File(dst, "r+") as f:
        for rod in sorted(f["cases"].keys(), key=lambda r: int(r[3:])):
            f["cases"][rod]["meas"][...] = noiseless[rod] * (1.0 + rel * rng.standard_normal())


def summarise(rep_dir: Path) -> dict:
    cal = rep_dir / "calibration"
    sc = load_results(sorted(cal.glob("SC_*.h5"))[-1])
    hc = load_results(sorted(cal.glob("HC_[0-9]*[0-9].h5"))[-1])
    names = list(sc["metadata"]["param_names"])
    idx = np.asarray(sc["metadata"]["calibrated_idx"])
    m, s = sc["scaler"]["mean"][idx], sc["scaler"]["std"][idx]
    b = int(sc["metadata"]["burnin"])
    th = sc["chains"]["theta"][:, b:, :].reshape(-1, len(names)) * s + m
    bh = int(hc["metadata"]["burnin"])
    mu = hc["chains"]["mu"][:, bh:, :].reshape(-1, len(names)) * s + m
    sig = hc["chains"]["sigma"][:, bh:, :].reshape(-1, len(names)) * s
    rng = np.random.default_rng(0)
    pop = mu + np.abs(sig) * rng.standard_normal(mu.shape)
    out = {}
    for j, n in enumerate(names):
        t = TRUE_A[n]
        for tag, x in (("SC", th[:, j]), ("HC_mu", mu[:, j]), ("HC_pop", pop[:, j])):
            lo, hi = np.percentile(x, [2.5, 97.5])
            out.update({f"{tag}_{n}_mean": float(x.mean()), f"{tag}_{n}_lo": float(lo),
                        f"{tag}_{n}_hi": float(hi), f"{tag}_{n}_cov": bool(lo <= t <= hi)})
        out[f"HC_sigma_{n}_mean"] = float(sig[:, j].mean())
    return out


def one_rep(r: int, args, base_cfg: dict, out: Path, noiseless: pd.Series, src: Path) -> str:
    rep_dir = out / f"rep_{r:03d}"
    done = rep_dir / "summary.json"
    if done.exists():
        return f"rep {r:03d}: already done"
    rep_dir.mkdir(parents=True, exist_ok=True)
    seed = args.seed0 + r
    make_data(src, rep_dir / "input_data.h5", seed, noiseless, args.noise)
    cfg = json.loads(json.dumps(base_cfg))
    cfg["project"]["output_dir"] = str(rep_dir)
    cfg["project"]["seed"] = seed
    cfg["data"]["input_file"] = str(rep_dir / "input_data.h5")
    cfg["hierarchical"]["nsteps_list"] = [args.hc_steps]
    cfg["hierarchical"]["parallel_chains"] = True
    cfg["standard"]["nsteps_list"] = [args.sc_steps]
    cfg["standard"]["stop_at_first_pass"] = True
    n_cal = sum(1 for p in cfg["parameters"].values() if p["calibrate"])
    cfg["standard"]["walker_multipliers"] = [max(1, args.sc_walkers // n_cal)]
    cpath = rep_dir / "config.yaml"
    cpath.write_text(yaml.safe_dump(cfg, sort_keys=False))
    with open(rep_dir / "log.txt", "w") as log:
        for step in ("04_run_standard", "05_run_hierarchical"):
            res = subprocess.run([sys.executable, "-u", str(WF / f"{step}.py"), str(cpath), "--no-progress"],
                                 cwd=HERE, stdout=log, stderr=subprocess.STDOUT, env=ENV)
            if res.returncode != 0:
                return f"rep {r:03d}: FAILED in {step} (see {rep_dir / 'log.txt'})"
    summ = {"rep": r, "seed": seed, **summarise(rep_dir)}
    done.write_text(json.dumps(summ, indent=1))
    if args.delete_chains:
        for f in (rep_dir / "calibration").glob("*.h5"):
            f.unlink()
    k = "UO2_k"
    return (f"rep {r:03d}: k  SC [{summ[f'SC_{k}_lo']:.3f}, {summ[f'SC_{k}_hi']:.3f}]  "
            f"HC pop [{summ[f'HC_pop_{k}_lo']:.3f}, {summ[f'HC_pop_{k}_hi']:.3f}]  (true {TRUE_A[k]:.3f})")


def coverage(out: Path):
    rows = [json.loads(f.read_text()) for f in sorted(out.glob("rep_*/summary.json"))]
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values("rep")
    df.to_csv(out / "repetitions.csv", index=False)
    cov = []
    for c in [c for c in df.columns if c.endswith("_cov")]:
        tag, name = c[:-4].split("_UO2_")[0], "UO2_" + c[:-4].split("_UO2_")[1]
        cov.append({"estimate": tag, "parameter": name, "covered": int(df[c].sum()), "n": len(df),
                    "coverage": float(df[c].mean()),
                    "mean_of_means": float(df[f"{tag}_{name}_mean"].mean()),
                    "mean_width": float((df[f"{tag}_{name}_hi"] - df[f"{tag}_{name}_lo"]).mean()),
                    "true": TRUE_A[name]})
    cov = pd.DataFrame(cov)
    cov.to_csv(out / "coverage.csv", index=False)
    print(f"\nCoverage of the true values over {len(df)} repetitions (95% intervals):")
    print(cov.round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", help="case name, e.g. modelbias_uniform_k or modelbias_struct_k")
    ap.add_argument("--n-cal", type=int, default=16, help="calibration rods (as the reference run)")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=15, help="repetitions in parallel (4 cores each)")
    ap.add_argument("--hc-steps", type=int, default=None, help="default: converged length of the reference run")
    ap.add_argument("--sc-steps", type=int, default=None)
    ap.add_argument("--sc-walkers", type=int, default=None)
    ap.add_argument("--seed0", type=int, default=50000)
    ap.add_argument("--noise", type=float, default=0.01, help="relative measurement noise")
    ap.add_argument("--quick", action="store_true", help="pipeline check on results_quick")
    ap.add_argument("--delete-chains", action="store_true")
    args = ap.parse_args()

    results = HERE / ("results_quick" if args.quick else "results")
    if None in (args.hc_steps, args.sc_steps, args.sc_walkers):
        hc, scn, scw = reference_lengths(results, args.case)
        args.hc_steps = args.hc_steps or hc
        args.sc_steps = args.sc_steps or scn
        args.sc_walkers = args.sc_walkers or scw
    cal_rods = calibration_subset(results, args.n_cal) if args.n_cal else None
    base = yaml.safe_load(effective_config(args.case, results, args.quick, cal_rods).read_text())
    src = Path(base["data"]["input_file"])
    exp = pd.read_csv(HERE / "inputs" / "synthetic_experiments.csv")
    noiseless = exp[exp["dataset"] == "A"].set_index("rod")["FCT_EOL_model_K"]

    out = results / f"repeat_{args.case}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "settings.json").write_text(json.dumps(vars(args) | {"calibration_rods": cal_rods}, indent=1))
    print(f"{args.case}: {args.reps} repetitions, {len(cal_rods or [])} calibration rods, "
          f"HC {args.hc_steps} steps x {base['hierarchical']['n_chains']} chains, "
          f"SC {args.sc_steps} steps x {args.sc_walkers} walkers, {args.jobs} in parallel -> {out}", flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for msg in pool.map(lambda r: one_rep(r, args, base, out, noiseless, src), range(args.reps)):
            print(msg, flush=True)
    coverage(out)


if __name__ == "__main__":
    main()

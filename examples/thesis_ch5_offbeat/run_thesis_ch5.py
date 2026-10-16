"""Chapter 5 of the thesis: the three scalar case studies (FCT at end of life, OFFBEAT,
130 rods derived from Turkey Point 3), standard vs hierarchical calibration.

    python run_thesis_ch5.py                    # everything (several hours: see README)
    python run_thesis_ch5.py prepare correlation morris kernels gp sobol
    python run_thesis_ch5.py epistemic aleatoric modelbias epistemic_prior
    python run_thesis_ch5.py figures
    python run_thesis_ch5.py all --quick        # pipeline check in a few minutes: 10 calibration
                                                # rods, short chains (NOT the thesis results)
    python run_thesis_ch5.py epistemic aleatoric modelbias epistemic_prior figures \
        --n-cal 16 --hc-steps 30000 50000       # reduced set: 6 validation + 16 calibration rods
                                                # (the next 16 of the correlation ranking)

Stages
  prepare      inputs/*.csv -> data/*.h5 (prepare_data.py)
  correlation  step 02: Spearman between rods (BH-FDR) -> 6 validation rods
  morris       Morris screening of 17 OFFBEAT multipliers (inputs/morris_tabular.dat, Dakota
               psuade_moat): mu*, sigma per rod, mean and 95% interval across rods (morris_ch5.py)
  kernels      step 03 --compare-kernels: 5 GP kernels compared on the test runs
  gp           step 03: one GP per rod (kernel of config_surrogates.yaml)
  sobol        Sobol' indices S1, ST of the 5 GP parameters with the prior uncertainty
               (literature widths, centred on the OFFBEAT default): which to calibrate (sobol_ch5.py)
  epistemic, aleatoric (k and alphaT calibrated),
  modelbias_uniform_k (k calibrated; alphaT, rho, relocation, swelling at the OFFBEAT default 1.0
               for every rod: bias common to all rods),
  modelbias_struct_k (k calibrated; alphaT at 1.0, rho, relocation, swelling linear in the rod
               power, make_modelbias_struct.py: bias dependent on the operating conditions),
  epistemic_prior (epistemic with literature normal priors: prior sensitivity)
               steps 04 (standard), 05 (hierarchical), 06 (diagnostics),
               07 (post-processing), 08 (validation: chi2 on the 6 validation rods)
               and export of 93 posterior samples per method (Wilks 95/95, two-sided)
  figures      thesis figures and summary tables (figures_ch5.py)

The configurations used are written to results/_configs/ (paths made absolute);
every output goes to results/ (results_quick/ with --quick).

Before running, the old outputs of the selected stages are deleted, so that a new run never
mixes with a previous one: with "all" (or "prepare") the whole data/ and results/ folders,
otherwise only the folders of the selected stages. --keep disables the deletion.

--continue-hc continues the last hierarchical chains of the selected cases instead of
restarting them (only for HC files written by this version, which store the chain
checkpoints), e.g. after a run at 50000 steps that did not converge:
    python run_thesis_ch5.py modelbias_struct_k --n-cal 16 --hc-steps 100000 --continue-hc
"""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
WF = HERE.parent.parent / "workflows"
CASES = ["epistemic", "aleatoric", "modelbias_uniform_k", "modelbias_struct_k", "epistemic_prior"]
STAGES = ["prepare", "correlation", "morris", "kernels", "gp", "sobol"] + CASES + ["figures"]
N_WILKS = 93
QUICK_N_CAL = 10


def _abs(value, base: Path):
    p = Path(value)
    return str(p if p.is_absolute() else (base / p).resolve())


def calibration_subset(results: Path, n_cal: int) -> list:
    """The first n_cal calibration rods of validation_split.csv (ranking by increasing
    maximum correlation with the other rods, i.e. the least redundant rods first)."""
    import pandas as pd
    f = results / "common" / "correlation" / "validation_split.csv"
    if not f.exists():
        sys.exit(f"--n-cal needs {f}: run the correlation stage first.")
    sp = pd.read_csv(f).sort_values("rank")
    rods = sp[sp["set"] == "calibration"]["case"].tolist()
    if n_cal > len(rods):
        sys.exit(f"--n-cal {n_cal}: only {len(rods)} calibration rods available.")
    return rods[:n_cal]


def effective_config(name: str, results: Path, quick: bool,
                     cal_rods: list | None = None, hc_steps: list | None = None) -> Path:
    """Copy of config_<name>.yaml with absolute paths (and quick-test / reduced-set overrides)."""
    cfg = yaml.safe_load((HERE / f"config_{name}.yaml").read_text())
    cfg = copy.deepcopy(cfg)
    out_rel = Path(cfg["project"]["output_dir"]).relative_to("results")
    cfg["project"]["output_dir"] = str(results / out_rel)
    d = cfg["data"]
    for key in ("input_file", "fixed_values_file"):
        if d.get(key):
            d[key] = _abs(d[key], HERE)
    if isinstance(d.get("validation_cases"), str):
        d["validation_cases"] = str(results / Path(d["validation_cases"]).relative_to("results"))
    if cfg.get("synthetic_truth_file"):
        cfg["synthetic_truth_file"] = _abs(cfg["synthetic_truth_file"], HERE)
    s = cfg.setdefault("surrogate", {})
    if s.get("file"):
        s["file"] = str(results / Path(s["file"]).relative_to("results"))
    if quick:
        _quick_overrides(name, cfg)
    if name != "surrogates":
        if cal_rods:
            cfg["data"]["calibration_cases"] = list(cal_rods)
        if hc_steps:
            cfg["hierarchical"]["nsteps_list"] = [int(n) for n in hc_steps]
    path = results / "_configs" / f"config_{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def _quick_overrides(name: str, cfg: dict):
    d = cfg["data"]
    d["input_file"] = d["input_file"].replace(".h5", "_quick.h5")
    if name == "surrogates":
        cfg["surrogate"]["n_restarts_optimizer"] = 1
        rods = _quick_rods()
        cfg["surrogate"]["kernel_comparison_cases"] = rods[:6]
        return
    cfg["standard"]["nsteps_list"] = [2000, 4000]
    cfg["hierarchical"]["nsteps_list"] = [2000, 4000]
    cfg["hierarchical"]["adaptation"]["proposal_update_start"] = 300
    cfg["postprocessing"]["n_posterior_samples"] = 500
    cfg["validation"]["n_draws"] = 500


def _quick_rods() -> list:
    """Quick test: 16 rods (6 + 10), the same in every stage."""
    return [f"ROD{i}" for i in (1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121)]


def _make_quick_data():
    """Subsets of the data files with the 16 rods of the quick test."""
    import h5py
    rods = _quick_rods()
    for ds in ("A", "B"):
        src, dst = HERE / "data" / f"input_data_{ds}.h5", HERE / "data" / f"input_data_{ds}_quick.h5"
        with h5py.File(src, "r") as f, h5py.File(dst, "w") as g:
            f.copy("inputs", g)
            gc = g.create_group("cases")
            for r in rods:
                f.copy(f"cases/{r}", gc)


def _use_recommended_kernel(common: Path, results: Path):
    """If the kernel comparison was run, train the GPs with the recommended kernel
    (lowest NLPD among the kernels with standardised-residual std in [0.8, 1.25]);
    otherwise keep surrogate.kernel of config_surrogates.yaml."""
    import pandas as pd
    f = results / "common" / "surrogates" / "kernel_comparison_summary.csv"
    cfg = yaml.safe_load(common.read_text())
    if not f.exists():
        print(f"  kernel of config_surrogates.yaml: {cfg['surrogate']['kernel']}")
        return
    best = pd.read_csv(f).query("recommended")["kernel"].iloc[0]
    spec = cfg["surrogate"]["kernel_candidates"][best]
    cfg["surrogate"]["kernel"] = {**cfg["surrogate"]["kernel"],
                                  **{k: v for k, v in spec.items() if k != "alpha_fixed"}}
    if spec.get("combine") is None and len(spec["terms"]) == 1:
        cfg["surrogate"]["kernel"]["combine"] = "sum"
    common.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"  kernel recommended by the comparison: {best} -> {cfg['surrogate']['kernel']}")


def check_consistency(results: Path):
    """GP of every rod at its true parameters vs the OFFBEAT synthetic run (no noise).
    The difference must be of the order of the GP error (~1%); a larger one means that the
    synthetic rods are not the rods of the training runs (power history, geometry, ...)."""
    import numpy as np
    import pandas as pd
    from abcf.surrogate import load_surrogates
    sset = load_surrogates(results / "common" / "surrogates" / "surrogates.pkl")
    exp = pd.read_csv(HERE / "inputs" / "synthetic_experiments.csv")
    names = sset.param_names
    rows = []
    for _, r in exp[exp["rod"].isin(list(sset.surrogates))].iterrows():
        m, v = sset[r["rod"]].predict(sset.scaler.transform(r[names].to_numpy(dtype=float)[None]))
        y = float(r["FCT_EOL_model_K"])
        rows.append({"dataset": r["dataset"], "rod": r["rod"], "offbeat_K": y, "gp_K": m[0, 0],
                     "rel_diff_pct": 100 * (m[0, 0] - y) / y, "gp_std_pct": 100 * np.sqrt(v[0, 0]) / y})
    df = pd.DataFrame(rows)
    df.to_csv(results / "common" / "surrogates" / "gp_vs_offbeat_at_true_values.csv", index=False)
    a = df["rel_diff_pct"].abs()
    print(f"  GP at the true values vs OFFBEAT synthetic runs: median |diff| {a.median():.2f}%, "
          f"max {a.max():.2f}% (GP std median {df['gp_std_pct'].median():.2f}%)")
    if a.median() > 3.0:
        sys.exit("  STOP: the synthetic experiments do not match the GP training runs (median "
                 "difference > 3%). Check that the synthetic rods use the same power history, "
                 "geometry and end time as the LHS runs.")


def run(cmd: list):
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    t0 = time.time()
    # -u: unbuffered output, so that the progress appears at once also when the output is
    # redirected to a file (e.g. | Tee-Object), not only at the end of each step
    res = subprocess.run([sys.executable, "-u"] + [str(c) for c in cmd], cwd=HERE)
    if res.returncode != 0:
        sys.exit(f"Failed: {' '.join(str(c) for c in cmd)}")
    print(f"  ({(time.time() - t0) / 60:.1f} min)", flush=True)


def clean_outputs(stages: list, results: Path):
    """Delete the old outputs of the stages that are about to run, so that a new run never
    mixes with the files of a previous one.

    prepare in the stages -> data/ and the whole results folder (everything is regenerated);
    otherwise only the folders written by the selected stages (the kernel comparison is kept
    when only the GPs are retrained, because the gp stage reads its recommendation)."""
    import shutil

    def rm(p: Path):
        if p.is_dir():
            shutil.rmtree(p)
            print(f"  removed {p.relative_to(HERE)}/")
        elif p.exists():
            p.unlink()

    print("\n=== clean: old outputs of the selected stages")
    if "prepare" in stages:
        rm(HERE / "data")
        rm(results)
        return
    sur = results / "common" / "surrogates"
    folders = {"correlation": [results / "common" / "correlation"],
               "morris": [results / "common" / "morris"],
               "kernels": [sur],
               "sobol": [results / "common" / "sobol"],
               "figures": [results / "figures"]}
    folders.update({c: [results / c, results / "_configs" / f"config_{c}.yaml"] for c in CASES})
    for s in stages:
        for p in folders.get(s, []):
            rm(p)
    if "gp" in stages and "kernels" not in stages and sur.is_dir():
        for p in list(sur.glob("*")) + list(sur.glob("figs/*")):
            if p.is_file() and not p.name.startswith("kernel_comparison"):
                p.unlink()
        print(f"  removed {sur.relative_to(HERE)}/ (kernel comparison kept)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="*", default=["all"], help=f"all or some of {STAGES}")
    ap.add_argument("--quick", action="store_true", help="fast pipeline check (not the thesis results)")
    ap.add_argument("--n-cal", type=int, default=None,
                    help="calibrate on the first N calibration rods of validation_split.csv only "
                         "(default: all 124); the 6 validation rods are unchanged")
    ap.add_argument("--hc-steps", type=int, nargs="+", default=None,
                    help="hierarchical chain lengths (overrides hierarchical.nsteps_list), "
                         "e.g. --hc-steps 30000 50000")
    ap.add_argument("--keep", action="store_true",
                    help="do not delete the old outputs of the selected stages before running them")
    ap.add_argument("--continue-hc", action="store_true",
                    help="continue the last hierarchical chains of each selected case up to the "
                         "longer lengths of --hc-steps (e.g. 50000 -> --hc-steps 100000) instead of "
                         "starting again; implies --keep and skips step 04 when an SC result exists")
    args = ap.parse_args()
    if args.continue_hc:
        args.keep = True
    stages = STAGES if "all" in args.stages else args.stages
    bad = [s for s in stages if s not in STAGES]
    if bad:
        sys.exit(f"Unknown stage(s) {bad}; valid: all, {', '.join(STAGES)}")
    results = HERE / ("results_quick" if args.quick else "results")
    if not args.keep:
        clean_outputs(stages, results)

    if "prepare" in stages:
        print("\n=== prepare: CSV -> HDF5")
        run([HERE / "prepare_data.py"])
        if args.quick:
            _make_quick_data()
    common = effective_config("surrogates", results, args.quick)
    if args.quick:
        cfg = yaml.safe_load(common.read_text())
        cfg["sensitivity"]["correlation"]["n_validation"] = 6
        common.write_text(yaml.safe_dump(cfg, sort_keys=False))
    if "correlation" in stages:
        print("\n=== correlation between rods and validation split (step 02)")
        run([WF / "02_correlation_analysis.py", common])
        if args.quick:      # keep 6 validation + 10 calibration rods
            import pandas as pd
            f = results / "common" / "correlation" / "validation_split.csv"
            sp = pd.read_csv(f)
            keep = sp[sp["set"] == "validation"]["case"].tolist() + \
                sp[sp["set"] == "calibration"]["case"].tolist()[:QUICK_N_CAL]
            sp[sp["case"].isin(keep)].to_csv(f, index=False)
    if "morris" in stages:
        print("\n=== Morris screening of the 17 OFFBEAT multipliers")
        run([HERE / "morris_ch5.py", HERE / "inputs" / "morris_tabular.dat",
             "--out", results / "common" / "morris"])
    if "kernels" in stages:
        print("\n=== GP kernel comparison (step 03 --compare-kernels)")
        run([WF / "03_train_surrogates.py", common, "--compare-kernels"])
    if "gp" in stages:
        print("\n=== GP training, one GP per rod (step 03)")
        _use_recommended_kernel(common, results)
        run([WF / "03_train_surrogates.py", common])
        check_consistency(results)
    if "sobol" in stages:
        print("\n=== Sobol' indices of the 5 GP parameters (prior uncertainty)")
        run([HERE / "sobol_ch5.py", "--surrogates", results / "common" / "surrogates" / "surrogates.pkl",
             "--out", results / "common" / "sobol"])

    cal_rods = calibration_subset(results, args.n_cal) if args.n_cal else None
    if cal_rods:
        print(f"\n  calibration on {len(cal_rods)} rods: {', '.join(cal_rods)}")
        (results / "_configs").mkdir(parents=True, exist_ok=True)
        (results / "_configs" / "calibration_rods.txt").write_text("\n".join(cal_rods) + "\n")
    cfgs = {n: effective_config(n, results, args.quick, cal_rods, args.hc_steps) for n in CASES}
    for name in CASES:
        if name not in stages:
            continue
        cfg = cfgs[name]
        print(f"\n=== {name.upper()} case")
        cal = results / name / "calibration"
        if args.continue_hc and sorted(cal.glob("SC_*.h5")):
            print(f"  standard calibration kept: {sorted(cal.glob('SC_*.h5'))[-1].name}")
        else:
            run([WF / "04_run_standard.py", cfg, "--no-progress"])
        if args.continue_hc and sorted(cal.glob("HC_*.h5")):
            run([WF / "05_run_hierarchical.py", cfg, "--no-progress", "--continue-from", "latest"])
        else:
            run([WF / "05_run_hierarchical.py", cfg, "--no-progress"])
        for step in ("06_diagnostics", "07_postprocessing", "08_validation"):
            run([WF / f"{step}.py", cfg])
        run([WF / "08_validation.py", cfg, "--export-samples", N_WILKS])

    if "figures" in stages:
        print("\n=== thesis figures and tables")
        sys.path.insert(0, str(HERE))
        from figures_ch5 import make_all
        make_all({n: cfgs[n] for n in CASES}, results / "figures")
    print("\nDone.")


if __name__ == "__main__":
    main()

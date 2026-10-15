"""Chapter 5 of the thesis: TRANSURANUS / Halden (IFPE) time-dependent case study with the
same pipeline as the OFFBEAT synthetic cases (examples/thesis_ch5_offbeat).

The IFPE measurements are not redistributable: the data are read from the user's own
results.h5 (stage prepare, --h5) and never committed. --facsimile runs everything on dummy
data with the same layout (make_facsimile.py), to test the pipeline without the IFPE data.

    python run_thesis_tu.py prepare --h5 <path>/results.h5     # -> data/input_data.h5
    python run_thesis_tu.py correlation kernels gp sobol         # common part
    python run_thesis_tu.py calibration                          # steps 04-08 + 93 samples
    python run_thesis_tu.py all --h5 <path>/results.h5           # everything
    python run_thesis_tu.py all --facsimile --quick              # pipeline check on dummy data

Stages
  prepare      results.h5 -> data/input_data.h5 (steady-state points >= 400 K, prepare_data.py)
  correlation  step 02: Spearman parameters vs PCA scores (BH-FDR) and between cases
               -> 6 validation cases (results/common/correlation/validation_split.csv)
  kernels      step 03 --compare-kernels: 5 GP kernels compared on the test runs
  gp           step 03: PCA (99% of the variance) + one GP per component and case
  sobol        Sobol' indices of the 5 multipliers on the PCA scores (per component and
               variance-weighted over the components): which to calibrate
  calibration  steps 04 (standard), 05 (hierarchical), 06 (diagnostics), 07 (post-processing),
               08 (validation through the GPs of the 6 validation cases) and export of 93
               posterior samples per method (Wilks 95/95, two-sided) to re-run TRANSURANUS

Before running, the old outputs of the selected stages are deleted (--keep disables it).
--continue-hc continues the last hierarchical chains up to the longer lengths of --hc-steps.
"""

import argparse
import copy
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
WF = HERE.parent.parent / "workflows"
STAGES = ["prepare", "correlation", "kernels", "gp", "sobol", "calibration"]
N_WILKS = 93


def run(cmd: list):
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    t0 = time.time()
    res = subprocess.run([sys.executable, "-u"] + [str(c) for c in cmd], cwd=HERE)
    if res.returncode != 0:
        sys.exit(f"Failed: {' '.join(str(c) for c in cmd)}")
    print(f"  ({(time.time() - t0) / 60:.1f} min)", flush=True)


def _abs(value, base: Path) -> str:
    p = Path(value)
    return str(p if p.is_absolute() else (base / p).resolve())


def effective_config(name: str, results: Path, input_file: Path, quick: bool,
                     cal_cases: list | None = None, hc_steps: list | None = None) -> Path:
    """Copy of config_<name>.yaml with absolute paths and the chosen data/results folders."""
    cfg = copy.deepcopy(yaml.safe_load((HERE / f"config_{name}.yaml").read_text()))
    cfg["project"]["output_dir"] = str(results / Path(cfg["project"]["output_dir"]).relative_to("results"))
    cfg["data"]["input_file"] = str(input_file)
    if isinstance(cfg["data"].get("validation_cases"), str):
        cfg["data"]["validation_cases"] = str(results / Path(cfg["data"]["validation_cases"]).relative_to("results"))
    s = cfg.setdefault("surrogate", {})
    if s.get("file"):
        s["file"] = str(results / Path(s["file"]).relative_to("results"))
    if quick:
        if name == "surrogates":
            s["n_restarts_optimizer"] = 1
        else:
            cfg["standard"]["nsteps_list"] = [2000, 4000]
            cfg["hierarchical"]["nsteps_list"] = [2000, 4000]
            cfg["hierarchical"]["adaptation"]["proposal_update_start"] = 300
            cfg["postprocessing"]["n_posterior_samples"] = 500
            cfg["validation"]["n_draws"] = 500
    if name != "surrogates":
        if cal_cases:
            cfg["data"]["calibration_cases"] = list(cal_cases)
        if hc_steps:
            cfg["hierarchical"]["nsteps_list"] = [int(n) for n in hc_steps]
    path = results / "_configs" / f"config_{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def calibration_subset(results: Path, n_cal: int) -> list:
    """The first n_cal calibration cases of validation_split.csv (least redundant first)."""
    import pandas as pd
    f = results / "common" / "correlation" / "validation_split.csv"
    if not f.exists():
        sys.exit(f"--n-cal needs {f}: run the correlation stage first.")
    sp = pd.read_csv(f).sort_values("rank")
    cases = sp[sp["set"] == "calibration"]["case"].tolist()
    if n_cal > len(cases):
        sys.exit(f"--n-cal {n_cal}: only {len(cases)} calibration cases available.")
    return cases[:n_cal]


def use_recommended_kernel(common: Path, results: Path):
    """If the kernel comparison was run, train the GPs with the recommended kernel (lowest
    NLPD among the kernels with standardised-residual std in [0.8, 1.25]); otherwise keep
    surrogate.kernel of config_surrogates.yaml (same rule as the OFFBEAT case studies)."""
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


def clean(stages: list, results: Path, data_file: Path):
    def rm(p: Path):
        if p.is_dir():
            shutil.rmtree(p)
            print(f"  removed {p}")
        elif p.exists():
            p.unlink()
    print("\n=== clean: old outputs of the selected stages")
    if "prepare" in stages:
        rm(results)
        return
    sur = results / "common" / "surrogates"
    folders = {"correlation": [results / "common" / "correlation"], "kernels": [sur],
               "sobol": [results / "common" / "sobol"], "calibration": [results / "tu"]}
    for s in stages:
        for p in folders.get(s, []):
            rm(p)
    if "gp" in stages and "kernels" not in stages and sur.is_dir():
        for p in list(sur.glob("*")) + list(sur.glob("figs/*")):
            if p.is_file() and not p.name.startswith("kernel_comparison"):
                p.unlink()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="*", default=["all"], help=f"all or some of {STAGES}")
    ap.add_argument("--h5", default=None, help="results.h5 of the TRANSURANUS-Dakota runs (stage prepare)")
    ap.add_argument("--facsimile", action="store_true", help="dummy data (make_facsimile.py), results_facsimile/")
    ap.add_argument("--quick", action="store_true", help="short chains (pipeline check, not thesis results)")
    ap.add_argument("--n-cal", type=int, default=None, help="calibrate on the first N calibration cases only")
    ap.add_argument("--hc-steps", type=int, nargs="+", default=None, help="hierarchical lengths")
    ap.add_argument("--keep", action="store_true", help="do not delete the old outputs")
    ap.add_argument("--continue-hc", action="store_true",
                    help="continue the last hierarchical chains (implies --keep, keeps the SC result)")
    args = ap.parse_args()
    stages = STAGES if "all" in args.stages else args.stages
    bad = [s for s in stages if s not in STAGES]
    if bad:
        sys.exit(f"Unknown stage(s) {bad}; valid: all, {', '.join(STAGES)}")
    if args.continue_hc:
        args.keep = True

    if args.facsimile:
        results, data_file = HERE / "results_facsimile", HERE / "data" / "input_data_facsimile.h5"
    else:
        results, data_file = HERE / "results", HERE / "data" / "input_data.h5"
    if not args.keep:
        clean(stages, results, data_file)

    if "prepare" in stages:
        print("\n=== prepare: TRANSURANUS results -> framework input file")
        if args.facsimile:
            run([HERE / "make_facsimile.py"])
            src = HERE / "data" / "results_facsimile.h5"
        elif args.h5:
            src = Path(args.h5).resolve()
        else:
            sys.exit("stage prepare: give --h5 <path>/results.h5 (or --facsimile)")
        run([HERE / "prepare_data.py", "--h5", src, "--out", data_file])
    if not data_file.exists():
        sys.exit(f"{data_file} not found: run the stage prepare first.")

    common = effective_config("surrogates", results, data_file, args.quick)
    if "correlation" in stages:
        print("\n=== correlation: parameters vs PCA scores, between cases, validation split (step 02)")
        run([WF / "02_correlation_analysis.py", common])
    if "kernels" in stages:
        print("\n=== GP kernel comparison (step 03 --compare-kernels)")
        run([WF / "03_train_surrogates.py", common, "--compare-kernels"])
    if "gp" in stages:
        print("\n=== PCA + GP training (step 03)")
        use_recommended_kernel(common, results)
        run([WF / "03_train_surrogates.py", common])
    if "sobol" in stages:
        print("\n=== Sobol' indices of the 5 multipliers")
        run([HERE / "sobol_tu.py", "--surrogates", results / "common" / "surrogates" / "surrogates.pkl",
             "--out", results / "common" / "sobol", "--n-base", 256 if args.quick else 1024])

    if "calibration" in stages:
        cal = calibration_subset(results, args.n_cal) if args.n_cal else None
        cfg = effective_config("tu", results, data_file, args.quick, cal, args.hc_steps)
        print("\n=== calibration on the real data (steps 04-08)")
        out = results / "tu" / "calibration"
        if args.continue_hc and sorted(out.glob("SC_*.h5")):
            print(f"  standard calibration kept: {sorted(out.glob('SC_*.h5'))[-1].name}")
        else:
            run([WF / "04_run_standard.py", cfg, "--no-progress"])
        if args.continue_hc and sorted(out.glob("HC_*.h5")):
            run([WF / "05_run_hierarchical.py", cfg, "--no-progress", "--continue-from", "latest"])
        else:
            run([WF / "05_run_hierarchical.py", cfg, "--no-progress"])
        for step in ("06_diagnostics", "07_postprocessing", "08_validation"):
            run([WF / f"{step}.py", cfg])
        run([WF / "08_validation.py", cfg, "--export-samples", N_WILKS])
    print("\nDone.")


if __name__ == "__main__":
    main()

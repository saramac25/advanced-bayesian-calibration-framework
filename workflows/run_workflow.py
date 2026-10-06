"""Run several workflow steps in a row for any study (main entry point).

    python workflows/run_workflow.py my_study/config.yaml                 # all applicable steps
    python workflows/run_workflow.py my_study/config.yaml --steps 3-7     # only steps 03 to 07
    python workflows/run_workflow.py my_study/config.yaml --steps 4,6     # only steps 04 and 06

Without --steps, the steps that cannot run are skipped automatically:
  01 (Morris) only if sensitivity.morris.model is set in the config;
  08 (validation) only if validation.run is true.
The run stops at the first step that fails.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from abcf.config import load_config

STEPS = {1: "01_sensitivity_morris", 2: "02_correlation_analysis", 3: "03_train_surrogates",
         4: "04_run_standard", 5: "05_run_hierarchical", 6: "06_diagnostics",
         7: "07_postprocessing", 8: "08_validation"}


def parse_steps(text: str) -> list:
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    bad = [s for s in out if s not in STEPS]
    if bad:
        sys.exit(f"Unknown step(s): {bad}. Valid steps: 1-8.")
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="path to the YAML configuration file")
    ap.add_argument("--steps", default=None, help='e.g. "3-7" or "4,6" (default: all applicable)')
    ap.add_argument("--progress", action="store_true", help="show the MCMC progress bars")
    args = ap.parse_args()

    config = Path(args.config).resolve()
    cfg = load_config(config)
    if args.steps:
        steps = parse_steps(args.steps)
    else:
        steps = list(STEPS)
        if not cfg["sensitivity"]["morris"]["model"]:
            steps.remove(1)
            print("Step 01 skipped: sensitivity.morris.model is not set.")
        if not cfg["validation"]["run"]:
            steps.remove(8)
            print("Step 08 skipped: validation.run is false.")

    here = Path(__file__).resolve().parent
    for s in steps:
        print(f"\n=================== {STEPS[s]}", flush=True)
        cmd = [sys.executable, str(here / f"{STEPS[s]}.py"), str(config)]
        if s in (4, 5) and not args.progress:
            cmd.append("--no-progress")
        result = subprocess.run(cmd, cwd=config.parent)
        if result.returncode != 0:
            sys.exit(f"\nStep {STEPS[s]} failed (see the message above). Fix it and resume with "
                     f"--steps {s}-{max(steps)}.")
    print(f"\nDone: steps {', '.join(f'{s:02d}' for s in steps)}.")


if __name__ == "__main__":
    main()

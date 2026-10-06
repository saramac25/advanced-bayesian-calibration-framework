"""Run the whole workflow on one example folder (works on Windows, Linux, macOS).

    python examples/run_all.py toy_scalar
    python examples/run_all.py toy_time_dependent
"""

import subprocess
import sys
from pathlib import Path

STEPS = ["01_sensitivity_morris", "02_correlation_analysis", "03_train_surrogates",
         "04_run_standard", "05_run_hierarchical", "06_diagnostics", "07_postprocessing"]


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    here = Path(__file__).resolve().parent
    folder = here / sys.argv[1]
    if not (folder / "config.yaml").exists():
        sys.exit(f"{folder / 'config.yaml'} not found.")
    if not (folder / "data" / "input_data.h5").exists():
        print("Example data not found: generating them with toy_model.py")
        subprocess.run([sys.executable, str(here / "toy_model.py")], check=True)
    for step in STEPS:
        print(f"\n=================== {step}")
        cmd = [sys.executable, str(here.parent / "workflows" / f"{step}.py"), "config.yaml"]
        if step.startswith(("04", "05")):
            cmd.append("--no-progress")
        subprocess.run(cmd, cwd=folder, check=True)
    print(f"\nDone. Results in {folder / 'results'}")


if __name__ == "__main__":
    main()

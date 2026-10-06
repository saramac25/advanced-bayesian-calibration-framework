"""Run the whole workflow on one example folder (works on Windows, Linux, macOS).

    python examples/run_all.py toy_scalar
    python examples/run_all.py toy_time_dependent

Generates the example data if needed, then calls workflows/run_workflow.py
(steps 01-07) on the example configuration.
"""

import subprocess
import sys
from pathlib import Path


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
    subprocess.run([sys.executable, str(here.parent / "workflows" / "run_workflow.py"),
                    str(folder / "config.yaml"), "--steps", "1-7"], check=True)
    print(f"\nResults in {folder / 'results'}")


if __name__ == "__main__":
    main()

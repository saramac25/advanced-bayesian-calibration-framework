"""Build the framework input files of the Chapter 5 case studies from the CSV tables.

Inputs (inputs/, committed to the repository):
  lhs_inputs.csv            600 OFFBEAT runs: eval_id and the 5 parameter multipliers
                            (Dakota LHS in [0.5, 1.5], seed 4344)
  lhs_FCT_EOL.csv           fuel centreline temperature at end of life [K] of the 600 runs,
                            one column per rod (ROD1 ... ROD130); empty = the run of that rod
                            crashed (314 of 78 000, at most 13 per rod): excluded for that rod only
  synthetic_experiments.csv synthetic experiments: true parameters of every rod for dataset A
                            (epistemic: one true vector for all rods) and B (aleatoric: one true
                            vector per rod), OFFBEAT FCT at EOL without and with 1% noise
  metadata.json             seed, distributions of the true values, true vector A, values of the model-bias case

Outputs (data/, not committed: rebuilt by this script):
  input_data_A.h5, input_data_B.h5     docs/DATA_FORMAT.md; same training runs, measurements of A or B
  synthetic_truth_A.h5, synthetic_truth_B.h5   true parameters of every rod
  fixed_values_aleatoric.csv           true rho, relocation, swelling of every rod (dataset B)

    python prepare_data.py
"""

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from abcf.io import write_input_data

HERE = Path(__file__).resolve().parent
PARAMS = ["UO2_rho", "UO2_k", "UO2_alphaT", "UO2_epsilonRelocation", "UO2_epsilonSwelling"]
FIXED = ["UO2_rho", "UO2_epsilonRelocation", "UO2_epsilonSwelling"]   # not calibrated


def main():
    inp, out = HERE / "inputs", HERE / "data"
    out.mkdir(exist_ok=True)
    X = pd.read_csv(inp / "lhs_inputs.csv")
    Y = pd.read_csv(inp / "lhs_FCT_EOL.csv")
    exp = pd.read_csv(inp / "synthetic_experiments.csv")
    meta = json.loads((inp / "metadata.json").read_text())

    assert (X["eval_id"].values == Y["eval_id"].values).all(), "eval_id mismatch between the LHS tables"
    rods = [c for c in Y.columns if c != "eval_id"]
    inputs = X[["eval_id"] + PARAMS]
    if not exp["completed"].all():
        raise ValueError("synthetic_experiments.csv contains runs that did not reach the end time")

    for ds in ("A", "B"):
        e = exp[exp["dataset"] == ds].set_index("rod")
        missing = [r for r in rods if r not in e.index]
        if missing:
            raise ValueError(f"dataset {ds}: no synthetic experiment for {missing}")
        cases = {r: {"tim": [0.0], "meas": [float(e.loc[r, "FCT_EOL_exp_K"])],
                     "pred": Y[r].to_numpy(dtype=float)[:, None], "eval_ids": Y["eval_id"].to_numpy()}
                 for r in rods}
        write_input_data(out / f"input_data_{ds}.h5", inputs, cases)
        with h5py.File(out / f"synthetic_truth_{ds}.h5", "w") as f:
            g = f.create_group("cases")
            for r in rods:
                g.create_group(r).create_dataset("theta", data=e.loc[r, PARAMS].to_numpy(dtype=float))
        print(f"dataset {ds}: {len(rods)} rods, {len(inputs)} training runs -> "
              f"input_data_{ds}.h5, synthetic_truth_{ds}.h5")

    b = exp[exp["dataset"] == "B"][["rod"] + FIXED].rename(columns={"rod": "case"})
    b.to_csv(out / "fixed_values_aleatoric.csv", index=False, float_format="%.10g")
    print("fixed_values_aleatoric.csv: true rho, relocation, swelling of every rod (dataset B)")

    # consistency of the configurations with metadata.json
    print("\nValues to be found in the configurations:")
    print("  epistemic fixed_value (true A): " +
          ", ".join(f"{p} = {meta['true_A'][p]:.10g}" for p in FIXED))
    print("  model-bias fixed_value (OFFBEAT default): " +
          ", ".join(f"{p} = {meta['bias']['values'][p]:.10g}" for p in FIXED))
    a_csv = exp[exp["dataset"] == "A"][PARAMS].drop_duplicates()
    assert len(a_csv) == 1 and np.allclose(a_csv.to_numpy()[0], [meta["true_A"][p] for p in PARAMS],
                                           rtol=1e-6), "true vector A differs between CSV and metadata"


if __name__ == "__main__":
    main()

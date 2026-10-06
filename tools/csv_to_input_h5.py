"""Build input_data.h5 from CSV tables (typical for a scalar QoI).

Expected files:
  * one CSV of training runs per case, named <case_id>.csv, with one
    column per parameter, one column with the QoI and optionally a
    column `eval_id` (otherwise the row number, starting at 1, is used);
    all cases must share the same parameter values for a given eval_id;
  * one CSV of measurements with columns `case` and `meas`.

Usage:
  python tools/csv_to_input_h5.py --runs-dir runs_csv/ --qoi-column qoi \
      --measurements measurements.csv --output data/input_data.h5
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from abcf.io import write_input_data


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", required=True, help="folder with one <case_id>.csv per case")
    ap.add_argument("--qoi-column", required=True, help="name of the QoI column in the run CSVs")
    ap.add_argument("--measurements", required=True, help="CSV with columns case, meas")
    ap.add_argument("--output", required=True, help="input_data.h5 to create")
    args = ap.parse_args()

    meas = pd.read_csv(args.measurements).set_index("case")["meas"]
    inputs, cases = None, {}
    for f in sorted(Path(args.runs_dir).glob("*.csv")):
        cid = f.stem
        if cid not in meas.index:
            print(f"  {cid}: no measurement, skipped")
            continue
        df = pd.read_csv(f)
        if "eval_id" not in df.columns:
            df.insert(0, "eval_id", np.arange(1, len(df) + 1))
        params = df.drop(columns=[args.qoi_column]).set_index("eval_id")
        if inputs is None:
            inputs = params
        else:
            common = inputs.index.intersection(params.index)
            if not np.allclose(inputs.loc[common].to_numpy(), params.loc[common].to_numpy()):
                raise ValueError(f"{f}: parameter values differ from the other cases for the same eval_id.")
            inputs = pd.concat([inputs, params.loc[params.index.difference(inputs.index)]])
        cases[cid] = {"tim": [0.0], "meas": [float(meas[cid])],
                      "pred": df[args.qoi_column].to_numpy(dtype=float)[:, None],
                      "eval_ids": df["eval_id"].to_numpy()}
    if not cases:
        raise ValueError("No case with both runs and a measurement.")
    write_input_data(args.output, inputs.reset_index(), cases)
    print(f"Written {args.output}: {len(cases)} case(s), {len(inputs)} solver runs, "
          f"parameters {list(inputs.columns)}")


if __name__ == "__main__":
    main()

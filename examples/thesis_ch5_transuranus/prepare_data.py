"""Build the framework input file of the TRANSURANUS / Halden case (Chapter 5) from the
HDF5 file of the TRANSURANUS-Dakota runs (results.h5, formerly dakota_data/results.h5).

The measurements come from the IFPE database (OECD/NEA) and are NOT redistributable: the
source file and data/input_data.h5 stay on the user's computer (data/ is in .gitignore).

Source layout (written by the TRANSURANUS-Dakota driver):
  dakota_data                     structured table: eval_id and the parameters (columns 'cal_*')
  cases/<case>/pred      (n_eval, n_t)   TRANSURANUS fuel centreline temperature [K]
  cases/<case>/meas      (n_t,)          thermocouple measurement [K]
  cases/<case>/tim       (n_t,)          time
  cases/<case>/eval_ids  (n_eval,)
  cases/<case>/node      scalar bytes    axial node of the thermocouple ('mean' = rod average)
  cases/<case>/filter    (n_t,) bool     True = point to DISCARD (manual screening)
  cases/<case>/meas_file structured      measurement file; column 'state' (0 = steady state)

Points kept (as in the previous scripts, templib_TU.case._filter):
  measurement not NaN, state == 0 (steady state), measurement >= --t-min (400 K by default:
  thermocouple data at low temperature are not representative), filter == False.

    python prepare_data.py --h5 <path>/results.h5            # -> data/input_data.h5
    python prepare_data.py --h5 <path>/results.h5 --exclude 515-10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from abcf.io import write_input_data

HERE = Path(__file__).resolve().parent


def read_cases(h5: Path, t_min: float, exclude: list):
    with h5py.File(h5, "r") as f:
        dk = pd.DataFrame.from_records(f["dakota_data"][:])
        params = [c for c in dk.columns if "cal" in c.lower()]
        inputs = dk[["eval_id"] + params].copy()
        cases, rows = {}, []
        for name, g in f["cases"].items():
            if any(p in name for p in exclude):
                continue
            meas = np.asarray(g["meas"][:], float).ravel()
            tim = np.asarray(g["tim"][:], float).ravel()
            pred = np.asarray(g["pred"][:], float)
            filt = np.asarray(g["filter"][:], bool).ravel()
            state = pd.DataFrame(g["meas_file"][:])["state"].astype(int).to_numpy()
            mask = ~np.isnan(meas) & (state == 0) & ~filt
            mask &= np.where(np.isnan(meas), False, meas >= t_min)
            ok_runs = ~np.isnan(pred[:, mask]).any(axis=1)          # failed runs: whole row dropped
            cases[name] = {"tim": tim[mask], "meas": meas[mask], "pred": pred[ok_runs][:, mask],
                           "eval_ids": np.asarray(g["eval_ids"][:]).ravel()[ok_runs]}
            node = g["node"][()]
            rows.append({"case": name, "node": node.decode() if isinstance(node, bytes) else str(node),
                         "n_points_total": meas.size, "n_points_kept": int(mask.sum()),
                         "n_runs": int(ok_runs.sum()), "t_first": float(tim[mask].min()),
                         "t_last": float(tim[mask].max()),
                         "meas_min_K": float(meas[mask].min()), "meas_max_K": float(meas[mask].max())})
    return inputs, cases, pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", required=True, help="results.h5 of the TRANSURANUS-Dakota runs")
    ap.add_argument("--out", default=str(HERE / "data" / "input_data.h5"))
    ap.add_argument("--t-min", type=float, default=400.0, help="discard measurements below this [K]")
    ap.add_argument("--exclude", nargs="*", default=[], help="skip the cases whose name contains these strings")
    args = ap.parse_args()
    inputs, cases, summary = read_cases(Path(args.h5), args.t_min, args.exclude)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_input_data(out, inputs, cases)
    summary.to_csv(out.with_name("cases_summary.csv"), index=False)
    print(f"{len(cases)} cases, {len(inputs)} runs, parameters {list(inputs.columns[1:])}")
    print(summary[["case", "node", "n_points_kept", "n_runs"]].to_string(index=False))
    print(f"Written {out} and {out.with_name('cases_summary.csv')} (not committed: IFPE data)")


if __name__ == "__main__":
    main()

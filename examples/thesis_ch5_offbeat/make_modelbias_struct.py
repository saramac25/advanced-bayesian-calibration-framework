"""Fixed values of the STRUCTURAL model-bias case (config_modelbias_struct_k.yaml).

The non-calibrated multipliers of the calibration model are a linear function of the
time-averaged linear heat generation rate (LHGR) of each rod, P_i [kW/m]:

    relocation_i = 1 + b_rel   (P_i - P_ref)
    swelling_i   = 1 + b_swell (P_i - P_ref)
    rho_i        = 1 + b_rho   (P_i - P_ref)          (clipped to the GP domain [0.5, 1.5])

with P_ref = 16.5 kW/m (median power of the calibration rods), b_rel = b_swell = 0.2 and
b_rho = 0.005 per kW/m; alphaT is fixed at the OFFBEAT default 1.0 in the configuration.
Reality (synthetic experiments, dataset A) has constant multipliers
(true vector A, inputs/metadata.json), so the model error of rod i,
delta_i = FCT(model multipliers of rod i) - FCT(true multipliers), has a part common to all
rods (multipliers 1.0 at P_ref instead of the true values) and a part that grows with the
rod power: a condition-dependent discrepancy delta(x), x = LHGR, in the sense of Kennedy and
O'Hagan (2001), introduced through the sub-models whose physics depends on power
(relocation, gaseous swelling, densification).

    python make_modelbias_struct.py --rod-dir <folder with ROD1 ... ROD130>/   (OFFBEAT cases)

writes inputs/fixed_values_modelbias_struct_k.csv (multipliers only). With zero slopes the
multipliers are 1.0 for every rod: the bias common to all rods of config_modelbias_uniform_k.yaml.
The power of every rod (rod_power.csv) is written next to this script only with --save-power,
and is not committed; it can be recovered from the density column of the output file,
P_i = P_ref + (rho_i - 1) / b_rho (never clipped), as done by figures_bias_ch5.py.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
P_REF, B_REL, B_SWELL, B_RHO = 16.5, 0.2, 0.2, 0.005


def mean_lhgr(case_dir: Path) -> float:
    """Time-averaged LHGR [kW/m] from constant/lhgr (timePoints [s], lhgr [W/m], trapezoid)."""
    s = (case_dir / "constant" / "lhgr").read_text()
    def block(key):
        return np.array(re.search(key + r"\s*\(([^)]*)\)", s).group(1).split(), dtype=float)
    t, q = block("timePoints"), block(r"\nlhgr")
    return float(np.trapezoid(q, t) / (t[-1] - t[0]) / 1e3)


def multipliers(p, b_rel=B_REL, b_swell=B_SWELL, b_rho=B_RHO):
    u = np.asarray(p, float) - P_REF
    clip = lambda v: np.clip(v, 0.5, 1.5)   # noqa: E731
    return clip(1 + b_rel * u), clip(1 + b_swell * u), clip(1 + b_rho * u)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rod-dir", required=True, help="folder with one OFFBEAT case per rod (ROD1 ...)")
    ap.add_argument("--save-power", action="store_true", help="also write rod_power.csv")
    ap.add_argument("--slope", type=float, default=B_REL,
                    help=f"b_rel = b_swell per kW/m (default {B_REL})")
    ap.add_argument("--rho-slope", type=float, default=B_RHO, help=f"b_rho per kW/m (default {B_RHO})")
    ap.add_argument("--out", default="fixed_values_modelbias_struct_k.csv", help="file name in inputs/")
    args = ap.parse_args()
    rods = sorted((d for d in Path(args.rod_dir).glob("ROD*") if (d / "constant" / "lhgr").exists()),
                  key=lambda d: int(d.name[3:]))
    p = pd.Series({d.name: mean_lhgr(d) for d in rods})
    rel, sw, rho = multipliers(p.values, args.slope, args.slope, args.rho_slope)
    out = pd.DataFrame({"case": p.index, "UO2_rho": rho.round(6), "UO2_epsilonRelocation": rel.round(6),
                        "UO2_epsilonSwelling": sw.round(6)})
    out.to_csv(HERE / "inputs" / args.out, index=False)
    if args.save_power:
        p.rename("lhgr_mean_kWm").rename_axis("case").to_csv(HERE / "rod_power.csv")
    print(f"{len(out)} rods: relocation {rel.min():.3f}-{rel.max():.3f}, swelling {sw.min():.3f}-{sw.max():.3f}, "
          f"rho {rho.min():.4f}-{rho.max():.4f} -> inputs/{args.out}")


if __name__ == "__main__":
    main()

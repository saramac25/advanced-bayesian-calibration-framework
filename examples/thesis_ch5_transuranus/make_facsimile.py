"""Facsimile of the TRANSURANUS-Dakota results.h5 with DUMMY data (no IFPE measurement).

Same layout as the real file (see prepare_data.py), so that the whole pipeline can be run
and tested without the IFPE data: 38 cases, 300 runs of 5 parameters (Latin hypercube, same
ranges as the real study), centreline temperatures from an analytical toy model of a fuel
rod under a random piecewise power history, measurements = toy model at a case-specific
"true" parameter vector + 1% noise. The numbers have no physical meaning.

    python make_facsimile.py            # -> data/results_facsimile.h5
"""

from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
PARAMS = ["cal_gapcon_f", "cal_lambda_a1", "cal_lambda_b0", "cal_lambda_b2", "cal_delg_f"]
RANGES = {"cal_gapcon_f": (0.8, 1.2), "cal_lambda_a1": (0.5, 1.5), "cal_lambda_b0": (0.5, 1.5),
          "cal_lambda_b2": (0.5, 1.5), "cal_delg_f": (0.5, 1.5)}


def toy_temperature(theta, lhgr, bu):
    """theta: (n, 5) multipliers; lhgr (n_t,) kW/m; bu (n_t,) burnup-like 0-1 -> (n, n_t) K."""
    g, a1, b0, b2, dg = (theta[:, [j]] for j in range(5))
    cond = 1.0 / (0.55 + 0.30 * a1 + 0.15 * b0 * (1 + 0.6 * bu) + 0.03 * b2 * bu ** 2)
    gap = 18.0 / (g * (1 + 0.8 * bu)) * (1 + 0.04 * (dg - 1) * bu)
    return 560.0 + lhgr * (gap + 30.0 * cond)


def main():
    rng = np.random.default_rng(12345)
    n_run = 300
    lhs = (rng.permuted(np.tile(np.arange(n_run), (5, 1)), axis=1).T + rng.uniform(size=(n_run, 5))) / n_run
    X = np.column_stack([RANGES[p][0] + (RANGES[p][1] - RANGES[p][0]) * lhs[:, j] for j, p in enumerate(PARAMS)])
    dt = np.dtype([("eval_id", "i8")] + [(p, "f8") for p in PARAMS])
    table = np.array([tuple([i + 1] + list(x)) for i, x in enumerate(X)], dtype=dt)
    nodes = ["3", "9", "10", "15", "mean"]
    out = HERE / "data" / "results_facsimile.h5"
    out.parent.mkdir(exist_ok=True)
    with h5py.File(out, "w") as f:
        f.create_dataset("dakota_data", data=table)
        gc = f.create_group("cases")
        for c in range(38):
            n_t = int(rng.integers(150, 1500))
            tim = np.sort(rng.uniform(50, 40000, n_t))
            steps = np.cumsum(rng.normal(0, 1.5, 12)) + rng.uniform(15, 30)
            lhgr = np.interp(tim, np.linspace(tim[0], tim[-1], 12), np.clip(steps, 5, 40))
            bu = (tim - tim[0]) / (tim[-1] - tim[0])
            pred = toy_temperature(X, lhgr, bu)
            pred[rng.integers(n_run)] = np.nan                                   # one failed run
            truth = np.array([1.0, 1.0, 1.0, 1.0, 1.0]) + rng.normal(0, [0.03, 0.08, 0.08, 0.05, 0.05])
            meas = toy_temperature(truth[None, :], lhgr, bu)[0] * (1 + 0.01 * rng.standard_normal(n_t))
            meas[rng.uniform(size=n_t) < 0.02] = np.nan
            state = (rng.uniform(size=n_t) < 0.05).astype(int)
            mf = np.zeros(n_t, dtype=[("time", "f8"), ("state", "i8")])
            mf["time"], mf["state"] = tim, state
            ok = ~np.isnan(pred).any(axis=1)
            g = gc.create_group(f"FAKE-{c // 4 + 1}-rod-{c % 4 + 1}-node-{nodes[c % 5]}")
            g.create_dataset("pred", data=pred[ok])
            g.create_dataset("eval_ids", data=np.arange(1, n_run + 1)[ok])
            g.create_dataset("meas", data=meas)
            g.create_dataset("tim", data=tim)
            g.create_dataset("filter", data=rng.uniform(size=n_t) < 0.01)
            g.create_dataset("meas_file", data=mf)
            g.create_dataset("node", data=np.bytes_(nodes[c % 5]))
    print(f"Written {out} (dummy data, 38 cases, {n_run} runs)")


if __name__ == "__main__":
    main()

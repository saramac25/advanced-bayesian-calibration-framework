"""Analytic toy model used by the examples and the tests.

QoI of case e at time t, parameters theta = (theta_1, theta_2, theta_3):

    y(t) = c_e * theta_1 * exp(-theta_2 * t) + 0.3 * theta_3 * t + 1

c_e is a case-specific operating condition. The scalar QoI is y at t = 1.
Nothing here is physical: it only exercises the whole workflow quickly.

    python examples/toy_model.py      -> writes examples/toy_scalar/ and examples/toy_time_dependent/
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from abcf.io import write_input_data
from abcf.model_interface import PythonModel

NAMES = ["theta_1", "theta_2", "theta_3"]
BOUNDS = np.array([[0.5, 1.5], [0.5, 1.5], [0.5, 1.5]])
CONDITIONS = {f"case_{i + 1:02d}": c for i, c in enumerate([0.8, 1.0, 1.2, 1.4, 1.6, 1.8])}
T_GRID = np.linspace(0.0, 3.0, 25)


def qoi(theta, case_id, scalar):
    t = np.array([1.0]) if scalar else T_GRID
    c = CONDITIONS[case_id]
    return c * theta[0] * np.exp(-theta[1] * t) + 0.3 * theta[2] * t + 1.0


class ToyModel(PythonModel):
    """ModelInterface for the Morris step (scalar QoI)."""

    def __init__(self):
        super().__init__(lambda th, cid: qoi([th[n] for n in NAMES], cid, scalar=True))


def make_example(folder: Path, scalar: bool, n_train: int = 80, seed: int = 1):
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "data").mkdir(exist_ok=True)

    X = qmc.scale(qmc.LatinHypercube(d=3, seed=seed).random(n_train), BOUNDS[:, 0], BOUNDS[:, 1])
    inputs = pd.DataFrame(X, columns=NAMES)
    inputs.insert(0, "eval_id", np.arange(1, n_train + 1))

    # True values: population N(mu, sigma); theta_3 fixed at 1
    # (scalar QoI: one measurement per case, so only theta_1 varies between cases;
    #  theta_2 is the same for all cases and is fixed in the configuration)
    mu_true = np.array([1.0, 0.9, 1.0])
    sig_true = np.array([0.12, 0.0, 0.0]) if scalar else np.array([0.12, 0.08, 0.0])
    cases, truth = {}, {}
    for cid in CONDITIONS:
        th = np.clip(mu_true + sig_true * rng.standard_normal(3), BOUNDS[:, 0] + 0.05, BOUNDS[:, 1] - 0.05)
        th[2] = 1.0
        y_true = qoi(th, cid, scalar)
        meas = y_true * (1 + 0.01 * rng.standard_normal(y_true.shape))
        pred = np.vstack([qoi(x, cid, scalar) for x in X])
        cases[cid] = {"tim": np.array([1.0]) if scalar else T_GRID, "meas": meas, "pred": pred,
                      "eval_ids": inputs["eval_id"].to_numpy()}
        truth[cid] = th
    write_input_data(folder / "data" / "input_data.h5", inputs, cases)

    import h5py
    with h5py.File(folder / "data" / "synthetic_truth.h5", "w") as f:
        g = f.create_group("cases")
        for cid, th in truth.items():
            g.create_group(cid).create_dataset("theta", data=th)


if __name__ == "__main__":
    here = Path(__file__).parent
    make_example(here / "toy_scalar", scalar=True)
    make_example(here / "toy_time_dependent", scalar=False)
    print("Example data written in examples/toy_scalar/data and examples/toy_time_dependent/data")

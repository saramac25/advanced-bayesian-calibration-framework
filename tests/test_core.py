"""Fast unit tests of the framework building blocks (pytest)."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from abcf.config import ParameterSetup, _default_config
from abcf.diagnostics.convergence import chain_diagnostics, mcse_obm
from abcf.io import DataFormatError, load_input_data, load_results, save_results, write_input_data
from abcf.sensitivity.morris import elementary_effects, morris_design, morris_indices
from abcf.surrogate.gp import CaseSurrogate, InputScaler, OutputReduction


def _toy_file(tmp_path, n_t=1, transpose=False):
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(30, 2))
    inputs = pd.DataFrame(X, columns=["a", "b"])
    inputs.insert(0, "eval_id", np.arange(1, 31))
    pred = np.column_stack([X[:, 0] + t * X[:, 1] for t in range(n_t)])
    cases = {"c1": {"tim": np.arange(n_t), "meas": pred[0], "pred": pred.T if transpose else pred,
                    "eval_ids": np.arange(1, 31)}}
    path = tmp_path / "input.h5"
    write_input_data(path, inputs, cases)
    return path


def test_input_roundtrip(tmp_path):
    data = load_input_data(_toy_file(tmp_path, n_t=5))
    c = data.cases["c1"]
    assert data.param_names == ["a", "b"]
    assert c.pred.shape == (30, 5) and c.n_t == 5
    assert data.X(c).shape == (30, 2)


def test_transposed_pred_is_rejected(tmp_path):
    import h5py
    path = _toy_file(tmp_path, n_t=5)
    with h5py.File(path, "a") as f:            # corrupt the file: transpose pred
        pred = f["cases/c1/pred"][()]
        del f["cases/c1/pred"]
        f["cases/c1/pred"] = pred.T
    with pytest.raises(DataFormatError):
        load_input_data(path)
    with pytest.raises(DataFormatError):
        _toy_file(tmp_path, n_t=5, transpose=True)


def test_results_roundtrip_never_overwrites(tmp_path):
    d = {"metadata": {"method": "SC", "names": ["a", "b"], "n": 3}, "chains": {"x": np.ones((2, 3))}}
    p1 = save_results(tmp_path / "r.h5", d)
    p2 = save_results(tmp_path / "r.h5", d)
    assert p1 != p2
    r = load_results(p1)
    assert r["metadata"]["names"] == ["a", "b"] and r["chains"]["x"].shape == (2, 3)


def test_default_config_loads():
    cfg = _default_config()
    assert cfg["hierarchical"]["adaptation"]["X"] in ("acceptance_rate", "covariance")
    assert isinstance(cfg["surrogate"]["kernel"]["length_scale_bounds"][1], float)


def test_parameter_setup_full_vector():
    s = ParameterSetup(["a", "b", "c"], [0, 2], {"b": 5.0})
    full = s.full_vector(np.array([[1.0, 3.0]]))
    assert np.allclose(full, [[1.0, 5.0, 3.0]])


def test_morris_linear_model():
    bounds = {"x1": [0, 1], "x2": [0, 2], "x3": [0, 1]}
    design = morris_design(bounds, num_trajectories=10, num_levels=4, seed=1)
    X = design.to_numpy()
    Y = 3 * X[:, 0] + 1 * X[:, 1] + 0 * X[:, 2]
    ee = elementary_effects(X, Y, np.array(list(bounds.values()), dtype=float))
    table = morris_indices(ee, list(bounds)).set_index("parameter")
    # EE on [0, 1]-rescaled inputs: slope * range
    assert np.isclose(table.loc["x1", "mu_star"], 3.0)
    assert np.isclose(table.loc["x2", "mu_star"], 2.0)
    assert np.isclose(table.loc["x3", "mu_star"], 0.0)


def test_pca_reduction_and_projection():
    rng = np.random.default_rng(1)
    Y = rng.normal(size=(50, 1)) @ rng.normal(size=(1, 20)) + 0.01 * rng.normal(size=(50, 20))
    red = OutputReduction(Y, 0.99)
    assert red.n_components <= 3
    assert np.allclose(red.reconstruct(red.project(Y)), Y, atol=0.1)
    scalar = OutputReduction(Y[:, :1], 0.99)
    assert scalar.n_components == 1 and np.allclose(scalar.project(Y[:, :1]), Y[:, :1])


def test_gp_variance_is_variance():
    rng = np.random.default_rng(2)
    X = rng.uniform(size=(25, 2))
    y = np.sin(3 * X[:, :1]) + X[:, 1:]
    cfg = _default_config()["surrogate"]
    cfg = dict(cfg, alpha_grid=None, n_restarts_optimizer=0)
    sur = CaseSurrogate("c").fit(X, y, cfg)
    x = X[:1] + 0.05
    _, var = sur.predict(x)
    _, std = sur.gps[0].predict(x, return_std=True)
    assert np.allclose(var[0, 0], std[0] ** 2)


def test_scalar_likelihood_matches_normal():
    from abcf.calibration.likelihood import CaseLikelihood, ParameterExpander
    rng = np.random.default_rng(3)
    X = rng.uniform(size=(30, 1))
    Y = 2 * X + 1
    cfg = dict(_default_config()["surrogate"], alpha_grid=None, n_restarts_optimizer=0)
    scaler = InputScaler(X)
    sur = CaseSurrogate("c").fit(scaler.transform(X), Y, cfg)
    setup = ParameterSetup(["a"], [0], {})
    lik = CaseLikelihood(sur, np.array([2.0]), 0.01, ParameterExpander(setup, scaler), False)
    xs = scaler.transform([[0.5]])
    pred = sur.predict(xs, return_var=False)[0, 0]
    assert np.isclose(lik(xs[0]), norm.logpdf(2.0, pred, 0.02))


def test_mcse_obm_iid():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(4, 20000))
    pooled = mcse_obm(x, pooled=True)
    assert abs(pooled - 1 / np.sqrt(80000)) / (1 / np.sqrt(80000)) < 0.2
    assert np.isclose(mcse_obm(x, pooled=False), pooled * 2, rtol=1e-12)
    d = chain_diagnostics(x[:, :, None])
    assert d["rhat"][0] < 1.01


def test_mhwg_sampler_normal_target():
    from abcf.calibration.hierarchical.mhwg.adaptation import adapt_covariance_matrix
    from abcf.calibration.hierarchical.mhwg.proposals import gauss_proposal
    from abcf.calibration.hierarchical.mhwg.random_walk import mh_sampler
    rng = np.random.default_rng(5)
    cov = np.array([[1.0, 0.8], [0.8, 1.0]])
    prec = np.linalg.inv(cov)
    smp = mh_sampler(gauss_proposal(np.eye(2), rng=rng), lambda x: -0.5 * x @ prec @ x,
                     rng=rng, adaptation=adapt_covariance_matrix(adapt_start=100,
                                                                 proposal_update_start=500,
                                                                 adapt_stop=5000))
    x = np.zeros(2)
    draws = []
    for i in range(30000):
        x = smp.step(x)
        if i >= 5000:
            draws.append(x)
    draws = np.array(draws)
    assert np.allclose(np.cov(draws.T), cov, atol=0.15)


# --------------------------------------------------------------------------- #
#  Additions for the thesis case studies
# --------------------------------------------------------------------------- #
def test_fast_predictor_matches_sklearn():
    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, size=(60, 3))
    y = (np.sin(2 * X[:, 0]) + X[:, 1] ** 2 + 0.1 * X[:, 2])[:, None]
    cfg_s = {"pca_variance": 0.99, "alpha_grid": None, "alpha_fixed": 1e-6, "n_restarts_optimizer": 0,
             "refit_on_bound": False,
             "kernel": {"terms": ["matern"], "combine": "sum", "matern_nu": 2.5, "white_noise": True,
                        "length_scale_bounds": [1e-3, 1e5]}}
    sur = CaseSurrogate("c").fit(X[:50], y[:50], cfg_s)
    m_ref, s_ref = sur.gps[0].predict(X[50:51], return_std=True)
    m, v = sur.predict(X[50:51])                       # fast path (1 point)
    assert np.allclose(m[0, 0], m_ref[0]) and np.allclose(v[0, 0], s_ref[0] ** 2, rtol=1e-8)


def test_case_specific_fixed_values():
    s = ParameterSetup(["a", "b", "c"], [0], {"b": 1.0, "c": 2.0}, {"r1": {"b": 5.0}})
    assert np.allclose(s.full_vector(np.array([[0.3]]), "r1"), [[0.3, 5.0, 2.0]])
    assert np.allclose(s.full_vector(np.array([[0.3]]), "r2"), [[0.3, 1.0, 2.0]])
    assert np.allclose(s.full_vector(np.array([[0.3]])), [[0.3, 1.0, 2.0]])


def test_validation_split_lowest_max_rho():
    from abcf.sensitivity.correlation import validation_split
    oo = pd.DataFrame({"case_A": ["a", "a", "b"], "case_B": ["b", "c", "c"], "output_index": [0, 0, 0],
                       "rho_squared": [0.98, 0.25, 0.36], "significant": [True, True, False]})
    sp = validation_split(oo, ["a", "b", "c"], 1)
    assert sp.loc[sp["set"] == "validation", "case"].tolist() == ["c"]   # max significant rho^2 0.25


def test_chi2_terms():
    from abcf.validation.metrics import chi2_terms
    t = chi2_terms([1000.0], [1050.0], [40.0], [5.0], [10.0])
    assert np.isclose(t["chi2_E_term"][0], 25.0)
    assert np.isclose(t["chi2_tot_term"][0], 2500 / (40 ** 2 + 5 ** 2 + 10 ** 2))

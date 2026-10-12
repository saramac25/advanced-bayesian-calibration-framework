"""Gaussian Process surrogates, one set per case.

Structure taken from ``gp_ensemble`` in Gustav Robertson's ``gp.py``:
one scikit-learn ``GaussianProcessRegressor`` per output component, the
nugget ``alpha`` chosen by cross-validation, outputs normalised. Added
here:

* the kernel is built from the configuration (RBF, Matern, rational
  quadratic, optional white noise, summed or multiplied), as in the
  scalar-QoI scripts;
* for a time-dependent QoI (``n_t > 1``) the outputs are first reduced
  with PCA, and one GP is trained per retained principal component;
* if the optimiser stops on a hyper-parameter bound, the GP is refitted
  with relaxed bounds;
* ``predict`` always returns the predictive VARIANCE explicitly, in the
  reduced (PCA) space where the likelihood is evaluated.

All GPs of a case work on standardised inputs (see ``InputScaler``).
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, ConstantKernel, Matern,
                                              RationalQuadratic, WhiteKernel)
from sklearn.model_selection import GridSearchCV
from scipy.linalg import solve_triangular


# --------------------------------------------------------------------------- #
#  Input standardisation shared by all cases
# --------------------------------------------------------------------------- #
class InputScaler:
    """Standardisation of the parameters: x_s = (x - mean) / std.

    Fitted once on all training inputs. Linear, so a Normal(mu, sigma) in
    standardised units maps to Normal(mean + std*mu, std*sigma) in
    physical units.
    """

    def __init__(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1.0
        Xs = self.transform(X)
        # Prior bounds = extent of the training data (in standardised units)
        self.lower = Xs.min(axis=0)
        self.upper = Xs.max(axis=0)

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean) / self.std

    def inverse(self, Xs):
        return np.asarray(Xs, dtype=float) * self.std + self.mean

    def to_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std, "lower": self.lower, "upper": self.upper}


# --------------------------------------------------------------------------- #
#  Kernel factory
# --------------------------------------------------------------------------- #
def build_kernel(n_features: int, kernel_cfg: dict, length_scale_bounds=None):
    """Kernel from the ``surrogate.kernel`` block of the configuration.

    amplitude * (term_1 (+ or *) term_2 ...) [+ white noise], with one
    length scale per input (ARD) for RBF and Matern.
    """
    lsb = tuple(float(v) for v in (length_scale_bounds or kernel_cfg["length_scale_bounds"]))
    terms = []
    for name in kernel_cfg["terms"]:
        if name == "rbf":
            terms.append(RBF(length_scale=np.ones(n_features), length_scale_bounds=lsb))
        elif name == "matern":
            terms.append(Matern(length_scale=np.ones(n_features), length_scale_bounds=lsb,
                                nu=float(kernel_cfg["matern_nu"])))
        elif name == "rational_quadratic":
            terms.append(RationalQuadratic(length_scale=1.0, alpha=1.0, length_scale_bounds=lsb))
        else:
            raise ValueError(f"Unknown kernel term: {name}")
    core = terms[0]
    for t in terms[1:]:
        core = core + t if kernel_cfg["combine"] == "sum" else core * t
    kernel = ConstantKernel(1.0, (1e-8, 1e5)) * core
    if kernel_cfg.get("white_noise"):
        kernel = kernel + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e0))
    return kernel


def _fit_one(Xs, y, cfg_s: dict, seed: int):
    """Fit one GP: alpha by CV, then refit with relaxed bounds if needed."""
    n_features = Xs.shape[1]

    def make(lsb, alpha):
        return GaussianProcessRegressor(
            kernel=build_kernel(n_features, cfg_s["kernel"], lsb),
            alpha=alpha, normalize_y=True,
            n_restarts_optimizer=cfg_s["n_restarts_optimizer"], random_state=seed)

    alpha = float(cfg_s["alpha_fixed"])
    if cfg_s.get("alpha_grid"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            search = GridSearchCV(make(None, alpha), {"alpha": [float(a) for a in cfg_s["alpha_grid"]]})
            search.fit(Xs, y)
        alpha = search.best_params_["alpha"]

    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always", ConvergenceWarning)
        gp = make(None, alpha).fit(Xs, y)
    hit_bound = any(issubclass(w.category, ConvergenceWarning) for w in wlist)
    if hit_bound and cfg_s.get("refit_on_bound"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp = make(cfg_s["relaxed_length_scale_bounds"], alpha).fit(Xs, y)
    return gp, alpha, hit_bound


class _FastPredictor:
    """Same predictive mean and variance as ``GaussianProcessRegressor.predict``
    (normalize_y=True), without its input checks: used for the single-point
    evaluations of the MCMC samplers, where those checks dominate the cost.
    Equality with scikit-learn is checked in tests/test_core.py."""

    def __init__(self, gp):
        self.kernel = gp.kernel_
        self.X_train = gp.X_train_
        self.alpha = np.ravel(gp.alpha_)
        L = gp.L_
        self.L_inv = solve_triangular(L, np.eye(L.shape[0]), lower=True)
        self.y_mean = float(np.ravel(gp._y_train_mean)[0])
        self.y_std = float(np.ravel(gp._y_train_std)[0])

    def __call__(self, X, return_var=True):
        K_trans = self.kernel(X, self.X_train)
        mean = K_trans @ self.alpha * self.y_std + self.y_mean
        if not return_var:
            return mean, None
        V = K_trans @ self.L_inv.T
        var = self.kernel.diag(X) - np.einsum("ij,ij->i", V, V)
        return mean, np.maximum(var, 0.0) * self.y_std ** 2


# --------------------------------------------------------------------------- #
#  Output reduction: PCA for n_t > 1, identity for a scalar QoI
# --------------------------------------------------------------------------- #
class OutputReduction:
    """z = (y - centre) @ W.T ; W has shape (k, n_t)."""

    def __init__(self, Y: np.ndarray, variance: float):
        n_t = Y.shape[1]
        if n_t == 1:
            self.pca = None
            self.center = np.zeros(1)
            self.W = np.ones((1, 1))
            self.explained_variance_ratio = np.ones(1)
        else:
            full = PCA().fit(Y)
            cum = np.cumsum(full.explained_variance_ratio_)
            k = int(np.searchsorted(cum, variance) + 1)
            k = min(k, len(cum))
            self.pca = PCA(n_components=k).fit(Y)
            self.center = self.pca.mean_
            self.W = self.pca.components_
            self.explained_variance_ratio = self.pca.explained_variance_ratio_

    @property
    def n_components(self) -> int:
        return self.W.shape[0]

    def project(self, Y):
        return (np.atleast_2d(Y) - self.center) @ self.W.T

    def reconstruct(self, Z):
        return np.atleast_2d(Z) @ self.W + self.center

    def project_covariance(self, cov_y):
        """Covariance in the reduced space: W Sigma_y W^T."""
        return self.W @ cov_y @ self.W.T


# --------------------------------------------------------------------------- #
#  Surrogate of one case
# --------------------------------------------------------------------------- #
class CaseSurrogate:
    """GP surrogate of one case on standardised inputs."""

    def __init__(self, case_id: str):
        self.case_id = case_id
        self.reduction = None
        self.gps = []
        self.alphas = []
        self.hit_bound = []
        self.test_metrics = {}

    def fit(self, Xs: np.ndarray, Y: np.ndarray, cfg_s: dict, seed: int = 0):
        self.reduction = OutputReduction(Y, cfg_s["pca_variance"])
        Z = self.reduction.project(Y)
        self.gps, self.alphas, self.hit_bound = [], [], []
        for k in range(Z.shape[1]):
            gp, alpha, hit = _fit_one(Xs, Z[:, k], cfg_s, seed + k)
            self.gps.append(gp)
            self.alphas.append(alpha)
            self.hit_bound.append(hit)
        return self

    @property
    def n_components(self) -> int:
        return len(self.gps)

    def _fast(self):
        fast = self.__dict__.get("_fast_predictors")
        if fast is None or len(fast) != len(self.gps):
            fast = [_FastPredictor(gp) for gp in self.gps]
            self.__dict__["_fast_predictors"] = fast
        return fast

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("_fast_predictors", None)      # rebuilt after loading
        return state

    def predict(self, Xs: np.ndarray, return_var: bool = True):
        """Mean and variance in the reduced space, each (n_points, k)."""
        Xs = np.atleast_2d(np.asarray(Xs, dtype=float))
        means, variances = [], []
        if Xs.shape[0] <= 16:          # MCMC: few points, fast path (same numbers)
            for f in self._fast():
                m, v = f(Xs, return_var)
                means.append(m)
                if return_var:
                    variances.append(v)
        else:
            for gp in self.gps:
                if return_var:
                    m, s = gp.predict(Xs, return_std=True)   # explicit keyword: std, then squared
                    variances.append(s ** 2)
                else:
                    m = gp.predict(Xs)
                means.append(m)
        mean = np.column_stack(means)
        if not return_var:
            return mean
        return mean, np.column_stack(variances)

    def predict_qoi(self, Xs: np.ndarray) -> np.ndarray:
        """Mean prediction of the QoI in the original space, (n_points, n_t)."""
        return self.reduction.reconstruct(self.predict(Xs, return_var=False))

    def evaluate(self, Xs_test, Y_test) -> dict:
        """R2 and RMSE on test runs (QoI space) and per component."""
        pred = self.predict_qoi(Xs_test)
        resid = Y_test - pred
        ss_res = np.sum(resid ** 2)
        ss_tot = np.sum((Y_test - Y_test.mean(axis=0)) ** 2)
        Z_test = self.reduction.project(Y_test)
        Z_pred, Z_var = self.predict(Xs_test)
        std_resid = (Z_test - Z_pred) / np.sqrt(np.maximum(Z_var, 1e-300))
        self.test_metrics = {
            "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "rmse": float(np.sqrt(np.mean(resid ** 2))),
            "r2_per_component": [float(1 - np.sum((Z_test[:, k] - Z_pred[:, k]) ** 2)
                                       / np.sum((Z_test[:, k] - Z_test[:, k].mean()) ** 2))
                                 for k in range(Z_test.shape[1])],
            # fraction of standardised residuals within +-1.96: ~0.95 if the GP variance is calibrated
            "coverage_95": float(np.mean(np.abs(std_resid) <= 1.96)),
        }
        return self.test_metrics


class SurrogateSet:
    """All case surrogates plus the shared input scaler."""

    def __init__(self, scaler: InputScaler, surrogates: dict, param_names: list):
        self.scaler = scaler
        self.surrogates = surrogates      # case_id -> CaseSurrogate
        self.param_names = list(param_names)

    def __getitem__(self, case_id):
        return self.surrogates[case_id]

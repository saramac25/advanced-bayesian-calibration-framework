"""Configuration loading and checking.

The user's YAML file is merged on top of the defaults shipped with the
package (``default_config.yaml``, identical to
``workflows/config_template.yaml``). Unknown keys are reported, so that
a typo does not silently fall back to a default.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np
import yaml


class ConfigError(ValueError):
    """Raised when the configuration is inconsistent."""


def _default_config() -> dict:
    text = resources.files("abcf").joinpath("default_config.yaml").read_text()
    cfg = yaml.safe_load(text)
    # The parameter block of the template is only an example.
    cfg["parameters"] = {}
    return cfg


def _merge(default: dict, user: dict, path: str = "") -> dict:
    out = copy.deepcopy(default)
    for key, value in (user or {}).items():
        where = f"{path}.{key}" if path else key
        if key not in default:
            # Free-form blocks: their keys are case ids or parameter names.
            if path not in ("parameters", "labels", "measurement_noise.per_case",
                            "hierarchical.fixed_proposal_cov", "validation.files",
                            "surrogate.kernel_candidates"):
                warnings.warn(f"Unknown configuration key '{where}' (ignored by the defaults "
                              "check, please verify the spelling).", stacklevel=3)
            out[key] = value
        elif isinstance(default[key], dict) and isinstance(value, dict):
            out[key] = _merge(default[key], value, where)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict:
    """Read a YAML configuration file and merge it with the defaults."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        user = yaml.safe_load(f) or {}
    cfg = _merge(_default_config(), user)
    cfg["_config_dir"] = str(path.resolve().parent)
    _check(cfg)
    return cfg


def normal_prior_std(cfg: dict, names: list, scaler, idx: list):
    """Normal priors of the calibrated parameters in standardised units.

    ``parameters.<name>.prior: {type: normal, mean: m, std: s}`` (physical units) ->
    (mean_s, std_s) arrays over the calibrated parameters, NaN where the prior is the
    default uniform over the training range. The normal prior is truncated to that range.
    Returns None if no calibrated parameter has a normal prior.
    """
    spec = cfg.get("parameters") or {}
    m = np.full(len(idx), np.nan)
    s = np.full(len(idx), np.nan)
    for j, k in enumerate(idx):
        pr = (spec.get(names[k]) or {}).get("prior") or {}
        if pr.get("type") == "normal":
            m[j] = (float(pr["mean"]) - scaler.mean[k]) / scaler.std[k]
            s[j] = float(pr["std"]) / scaler.std[k]
    return None if np.all(np.isnan(s)) else (m, s)


def normal_logprior(x, prior) -> float:
    """Sum of the normal log-priors (standardised units); 0 for uniform components."""
    if prior is None:
        return 0.0
    m, s = prior
    ok = ~np.isnan(s)
    z = (np.asarray(x, dtype=float)[ok] - m[ok]) / s[ok]
    return float(-0.5 * np.sum(z * z))


def param_label(cfg: dict, name: str) -> str:
    """Display name of a parameter in the figures (``labels`` block), or its name."""
    return (cfg.get("labels") or {}).get(name, name)


def resolve_path(cfg: dict, value: str | Path) -> Path:
    """Paths in the config are relative to the folder of the config file."""
    p = Path(value)
    return p if p.is_absolute() else Path(cfg["_config_dir"]) / p


def output_dir(cfg: dict, *sub: str) -> Path:
    """Output folder (created if needed). Never deletes anything."""
    d = resolve_path(cfg, cfg["project"]["output_dir"]).joinpath(*sub)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _check(cfg: dict) -> None:
    noise = cfg["measurement_noise"]
    if noise["relative"] is None or noise["relative"] <= 0:
        raise ConfigError("measurement_noise.relative must be a positive number.")
    for case, value in (noise.get("per_case") or {}).items():
        if value <= 0:
            raise ConfigError(f"measurement_noise.per_case.{case} must be positive.")

    for name, spec in cfg["parameters"].items():
        if not isinstance(spec, dict) or "calibrate" not in spec:
            raise ConfigError(f"parameters.{name}: write at least {{calibrate: true/false}}.")
        if not spec["calibrate"] and spec.get("fixed_value") is None:
            raise ConfigError(f"parameters.{name}: calibrate is false, so fixed_value is required.")
        pr = spec.get("prior")
        if pr is not None:
            if not spec["calibrate"]:
                raise ConfigError(f"parameters.{name}.prior: only calibrated parameters have a prior.")
            if pr.get("type") not in ("uniform", "normal"):
                raise ConfigError(f"parameters.{name}.prior.type must be 'uniform' or 'normal'.")
            if pr["type"] == "normal" and not (float(pr.get("std", 0)) > 0 and "mean" in pr):
                raise ConfigError(f"parameters.{name}.prior: a normal prior needs mean and std > 0.")
        if spec.get("bounds") is not None:
            lo, hi = spec["bounds"]
            if not lo < hi:
                raise ConfigError(f"parameters.{name}.bounds: lower bound must be < upper bound.")

    h = cfg["hierarchical"]
    if not 0 < h["burnin_fraction"] < 1:
        raise ConfigError("hierarchical.burnin_fraction must be in (0, 1).")
    for block in ("mu", "sigma", "X"):
        if h["adaptation"][block] not in ("acceptance_rate", "covariance"):
            raise ConfigError(f"hierarchical.adaptation.{block} must be 'acceptance_rate' or 'covariance'.")
    if "type" in h["adaptation"]:
        raise ConfigError("hierarchical.adaptation.type was replaced by one entry per block: "
                          "adaptation: {mu: ..., sigma: ..., X: ...}.")
    for n in (h.get("nsteps_list") or [h["nsteps"]]):
        if h["adaptation"]["start"] >= n * h["burnin_fraction"] - int(h["adaptation"]["stop_before_burnin_end"]):
            raise ConfigError(f"hierarchical.adaptation.start must come before the end of the adaptation "
                              f"(burn-in - stop_before_burnin_end) for nsteps = {n}.")

    s = cfg["standard"]
    if not 0 < s["burnin_fraction"] < 1:
        raise ConfigError("standard.burnin_fraction must be in (0, 1).")
    if s["stretch_a"] <= 1:
        raise ConfigError("standard.stretch_a must be > 1.")

    if cfg["postprocessing"]["mode"] not in ("standard", "hierarchical", "both"):
        raise ConfigError("postprocessing.mode must be standard, hierarchical or both.")

    k = cfg["surrogate"]["kernel"]
    allowed = {"rbf", "matern", "rational_quadratic"}
    if not k["terms"] or not set(k["terms"]) <= allowed:
        raise ConfigError(f"surrogate.kernel.terms must be a non-empty subset of {sorted(allowed)}.")
    if k["combine"] not in ("sum", "product"):
        raise ConfigError("surrogate.kernel.combine must be 'sum' or 'product'.")


@dataclass
class ParameterSetup:
    """Split of the model parameters into calibrated and fixed ones.

    ``case_fixed`` optionally overrides ``fixed_values`` for single cases
    (``data.fixed_values_file``): e.g. a non-calibrated parameter known to
    take a different value in every experiment.
    """

    names: list            # all parameter names, in the order of the data file
    calibrated_idx: list   # indices (into `names`) of the calibrated parameters
    fixed_values: dict     # name -> fixed value (physical units)
    case_fixed: dict = None  # case_id -> {name: fixed value} (physical units), optional

    @property
    def calibrated_names(self) -> list:
        return [self.names[i] for i in self.calibrated_idx]

    @property
    def n_calibrated(self) -> int:
        return len(self.calibrated_idx)

    def fixed_for(self, case_id=None) -> dict:
        """Fixed values of one case (case-specific values override the common ones)."""
        out = dict(self.fixed_values)
        if case_id is not None and self.case_fixed and case_id in self.case_fixed:
            out.update(self.case_fixed[case_id])
        return out

    def full_vector(self, calibrated_values: np.ndarray, case_id=None) -> np.ndarray:
        """Physical full parameter vector(s) from calibrated values (physical).

        With ``case_id``, the fixed parameters take the values of that case
        (``data.fixed_values_file``) when they are given.
        """
        calibrated_values = np.atleast_2d(calibrated_values)
        fixed = self.fixed_for(case_id)
        out = np.empty((calibrated_values.shape[0], len(self.names)))
        for j, name in enumerate(self.names):
            if j not in self.calibrated_idx:
                out[:, j] = fixed[name]
        out[:, self.calibrated_idx] = calibrated_values
        return out


def parameter_setup(cfg: dict, data_param_names: list) -> ParameterSetup:
    """Match the parameter block of the config with the parameters in the data."""
    spec = cfg["parameters"]
    if not spec:
        # No parameter block: calibrate everything.
        return ParameterSetup(list(data_param_names), list(range(len(data_param_names))), {})
    missing = [n for n in data_param_names if n not in spec]
    extra = [n for n in spec if n not in data_param_names]
    if missing or extra:
        raise ConfigError(
            "The parameters in the config do not match the columns of `inputs` in the data file.\n"
            f"  in the data but not in the config: {missing}\n"
            f"  in the config but not in the data: {extra}")
    calibrated = [i for i, n in enumerate(data_param_names) if spec[n]["calibrate"]]
    if not calibrated:
        raise ConfigError("At least one parameter must have calibrate: true.")
    fixed = {n: float(spec[n]["fixed_value"]) for n in data_param_names if not spec[n]["calibrate"]}
    return ParameterSetup(list(data_param_names), calibrated, fixed,
                          _case_fixed_values(cfg, [n for n in data_param_names if not spec[n]["calibrate"]]))


def _case_fixed_values(cfg: dict, fixed_names: list) -> dict | None:
    """Read ``data.fixed_values_file``: CSV with a column ``case`` and one column per
    non-calibrated parameter whose value changes from case to case (physical units).
    Parameters not listed keep their common ``fixed_value``."""
    f = cfg["data"].get("fixed_values_file")
    if not f:
        return None
    import pandas as pd
    path = resolve_path(cfg, f)
    if not path.exists():
        raise ConfigError(f"data.fixed_values_file not found: {path}")
    df = pd.read_csv(path)
    if "case" not in df.columns:
        raise ConfigError(f"{path}: a column `case` is required.")
    cols = [c for c in df.columns if c != "case"]
    wrong = [c for c in cols if c not in fixed_names]
    if wrong:
        raise ConfigError(f"{path}: columns {wrong} are not non-calibrated parameters "
                          f"(non-calibrated: {fixed_names}).")
    return {str(r["case"]): {c: float(r[c]) for c in cols} for _, r in df.iterrows()}


def validation_case_list(cfg: dict) -> list:
    """``data.validation_cases``: a list of case ids, or the path of a CSV with
    columns ``case`` and ``set`` (rows with set == "validation" are used), e.g. the
    split written by step 02 (``results/correlation/validation_split.csv``)."""
    v = cfg["data"]["validation_cases"]
    if not v:
        return []
    if isinstance(v, str):
        import pandas as pd
        path = resolve_path(cfg, v)
        if not path.exists():
            raise ConfigError(f"data.validation_cases: file not found: {path} "
                              "(run step 02 first if it is the split written by it).")
        df = pd.read_csv(path)
        return [str(c) for c in df.loc[df["set"] == "validation", "case"]]
    return [str(c) for c in v]


def case_noise(cfg: dict, case_id: str) -> float:
    """Relative measurement uncertainty of one case."""
    noise = cfg["measurement_noise"]
    return float((noise.get("per_case") or {}).get(case_id, noise["relative"]))

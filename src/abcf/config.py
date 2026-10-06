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
            if path not in ("parameters", "measurement_noise.per_case",
                            "hierarchical.fixed_proposal_cov", "validation.files"):
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
        if spec.get("bounds") is not None:
            lo, hi = spec["bounds"]
            if not lo < hi:
                raise ConfigError(f"parameters.{name}.bounds: lower bound must be < upper bound.")

    h = cfg["hierarchical"]
    if not 0 < h["burnin_fraction"] < 1:
        raise ConfigError("hierarchical.burnin_fraction must be in (0, 1).")
    if h["adaptation"]["type"] not in ("acceptance_rate", "covariance"):
        raise ConfigError("hierarchical.adaptation.type must be 'acceptance_rate' or 'covariance'.")
    if h["adaptation"]["start"] >= h["nsteps"] * h["burnin_fraction"]:
        raise ConfigError("hierarchical.adaptation.start must come before the end of the burn-in.")

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
    """Split of the model parameters into calibrated and fixed ones."""

    names: list            # all parameter names, in the order of the data file
    calibrated_idx: list   # indices (into `names`) of the calibrated parameters
    fixed_values: dict     # name -> fixed value (physical units)

    @property
    def calibrated_names(self) -> list:
        return [self.names[i] for i in self.calibrated_idx]

    @property
    def n_calibrated(self) -> int:
        return len(self.calibrated_idx)

    def full_vector(self, calibrated_values: np.ndarray) -> np.ndarray:
        """Physical full parameter vector(s) from calibrated values (physical)."""
        calibrated_values = np.atleast_2d(calibrated_values)
        out = np.empty((calibrated_values.shape[0], len(self.names)))
        for j, name in enumerate(self.names):
            if j not in self.calibrated_idx:
                out[:, j] = self.fixed_values[name]
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
    return ParameterSetup(list(data_param_names), calibrated, fixed)


def case_noise(cfg: dict, case_id: str) -> float:
    """Relative measurement uncertainty of one case."""
    noise = cfg["measurement_noise"]
    return float((noise.get("per_case") or {}).get(case_id, noise["relative"]))

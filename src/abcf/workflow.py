"""Helpers shared by the workflow scripts (loading, case selection, results files)."""

from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path

import numpy as np

from . import io
from .config import load_config, output_dir, parameter_setup, resolve_path, validation_case_list


def parse_args(description: str, extra=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("config", help="path to the YAML configuration file")
    if extra:
        extra(parser)
    return parser.parse_args()


def load_all(config_path, need_surrogates: bool = True):
    """Config, data, parameter setup and (optionally) surrogates."""
    cfg = load_config(config_path)
    data = io.load_input_data(resolve_path(cfg, cfg["data"]["input_file"]), "all")
    setup = parameter_setup(cfg, data.param_names)
    sset = None
    if need_surrogates:
        from .surrogate import load_surrogates
        path = surrogate_path(cfg)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: run workflows/03_train_surrogates.py first.")
        sset = load_surrogates(path)
    return cfg, data, setup, sset


def surrogate_path(cfg) -> Path:
    """Surrogate pickle: ``surrogate.file`` if set (shared between configurations),
    otherwise results/surrogates/surrogates.pkl."""
    f = cfg["surrogate"].get("file")
    if f:
        p = resolve_path(cfg, f)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    return output_dir(cfg, "surrogates") / "surrogates.pkl"


def calibration_cases(cfg, data) -> list:
    sel = cfg["data"]["calibration_cases"]
    ids = data.case_ids if sel == "all" else list(sel)
    val = set(validation_case_list(cfg))
    ids = [c for c in ids if c not in val]
    missing = [c for c in ids if c not in data.cases]
    if missing:
        raise ValueError(f"Calibration cases not found in the data: {missing}")
    if not ids:
        raise ValueError("No calibration case left (check calibration_cases / validation_cases).")
    return ids


def qoi_type(data, case_ids) -> str:
    return "scalar" if all(data.cases[c].is_scalar for c in case_ids) else "time_dependent"


def timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def latest_result(cfg, method: str) -> Path | None:
    """Most recent results file of a method ('SC' or 'HC')."""
    folder = output_dir(cfg, "calibration")
    files = sorted(folder.glob(f"{method}_*.h5"))
    return files[-1] if files else None


def to_physical(values_std: np.ndarray, scaler: dict, idx) -> np.ndarray:
    """Standardised calibrated values -> physical units (last axis = parameters)."""
    return values_std * scaler["std"][idx] + scaler["mean"][idx]


def sigma_to_physical(sigma_std: np.ndarray, scaler: dict, idx) -> np.ndarray:
    return sigma_std * scaler["std"][idx]

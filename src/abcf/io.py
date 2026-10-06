"""Reading and checking the input files, writing and reading results.

Formats are described in ``docs/DATA_FORMAT.md``. Every reader checks
the file before returning, and stops with a message that names the
case and the dataset at fault.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


class DataFormatError(ValueError):
    """Raised when an input file does not follow docs/DATA_FORMAT.md."""


@dataclass
class CaseData:
    """One experiment (case): measurement and training simulations."""

    case_id: str
    tim: np.ndarray        # (n_t,) time grid, filtered
    meas: np.ndarray       # (n_t,) measured QoI, filtered
    pred: np.ndarray       # (n_eval, n_t) simulated QoI, filtered
    eval_ids: np.ndarray   # (n_eval,)

    @property
    def n_t(self) -> int:
        return self.meas.shape[0]

    @property
    def is_scalar(self) -> bool:
        return self.n_t == 1


@dataclass
class InputData:
    """Content of input_data.h5."""

    inputs: pd.DataFrame                       # index eval_id, one column per parameter
    cases: dict = field(default_factory=dict)  # case_id -> CaseData

    @property
    def param_names(self) -> list:
        return list(self.inputs.columns)

    @property
    def case_ids(self) -> list:
        return list(self.cases)

    def X(self, case: CaseData) -> np.ndarray:
        """Parameter values (physical) of the training runs of a case, row-aligned with pred."""
        return self.inputs.loc[case.eval_ids].to_numpy(dtype=float)


# --------------------------------------------------------------------------- #
#  input_data.h5
# --------------------------------------------------------------------------- #
def _read_table(dset) -> pd.DataFrame:
    if dset.dtype.names is None:
        raise DataFormatError("`inputs` must be a structured (table) dataset with named columns.")
    df = pd.DataFrame.from_records(dset[()])
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda v: v.decode() if isinstance(v, bytes) else v)
    return df


def load_input_data(path: str | Path, case_ids: list | str = "all") -> InputData:
    """Read and validate input_data.h5.

    The optional ``filter`` mask of each case is applied here, so the rest
    of the framework only sees the points to be used.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with h5py.File(path, "r") as f:
        if "inputs" not in f:
            raise DataFormatError(f"{path}: missing table `inputs`.")
        if "cases" not in f or len(f["cases"]) == 0:
            raise DataFormatError(f"{path}: missing group `cases` (or it is empty).")

        inputs = _read_table(f["inputs"])
        if "eval_id" not in inputs.columns:
            raise DataFormatError(f"{path}: `inputs` must contain a column `eval_id`.")
        if inputs["eval_id"].duplicated().any():
            raise DataFormatError(f"{path}: `inputs/eval_id` contains duplicates.")
        inputs = inputs.set_index("eval_id")
        non_numeric = [c for c in inputs.columns if not np.issubdtype(inputs[c].dtype, np.number)]
        if non_numeric:
            raise DataFormatError(f"{path}: non-numeric parameter columns in `inputs`: {non_numeric}")
        if inputs.isna().any().any():
            raise DataFormatError(f"{path}: `inputs` contains NaN values.")

        wanted = list(f["cases"]) if case_ids == "all" else list(case_ids)
        missing = [c for c in wanted if c not in f["cases"]]
        if missing:
            raise DataFormatError(f"{path}: cases requested in the config but not in the file: {missing}")

        cases = {}
        for cid in wanted:
            g = f["cases"][cid]
            for key in ("tim", "meas", "pred", "eval_ids"):
                if key not in g:
                    raise DataFormatError(f"{path}: case `{cid}` has no dataset `{key}`.")
            tim = np.asarray(g["tim"][()], dtype=float).ravel()
            meas = np.asarray(g["meas"][()], dtype=float).ravel()
            pred = np.atleast_2d(np.asarray(g["pred"][()], dtype=float))
            eval_ids = np.asarray(g["eval_ids"][()]).ravel()
            n_t = meas.shape[0]
            if pred.shape[1] != n_t and pred.shape[0] == n_t and n_t > 1:
                raise DataFormatError(
                    f"{path}: case `{cid}`: `pred` has shape {pred.shape}; expected "
                    f"(n_eval, {n_t}) - it looks transposed.")
            if n_t == 1 and pred.shape[0] == 1 and pred.shape[1] > 1:
                pred = pred.T          # scalar QoI stored as a row vector
            if tim.shape[0] != n_t:
                raise DataFormatError(f"{path}: case `{cid}`: `tim` has {tim.shape[0]} points, "
                                      f"`meas` has {n_t}.")
            if pred.shape[1] != n_t:
                raise DataFormatError(f"{path}: case `{cid}`: `pred` has {pred.shape[1]} columns, "
                                      f"`meas` has {n_t} points.")
            if pred.shape[0] != eval_ids.shape[0]:
                raise DataFormatError(f"{path}: case `{cid}`: `pred` has {pred.shape[0]} rows, "
                                      f"`eval_ids` has {eval_ids.shape[0]} entries.")
            unknown = np.setdiff1d(eval_ids, inputs.index.to_numpy())
            if unknown.size:
                raise DataFormatError(f"{path}: case `{cid}`: eval_ids not found in `inputs`: "
                                      f"{unknown[:10].tolist()}{' ...' if unknown.size > 10 else ''}")

            mask = np.ones(n_t, dtype=bool)
            if "filter" in g:
                mask = np.asarray(g["filter"][()], dtype=bool).ravel()
                if mask.shape[0] != n_t:
                    raise DataFormatError(f"{path}: case `{cid}`: `filter` has {mask.shape[0]} "
                                          f"points, `meas` has {n_t}.")
            mask &= np.isfinite(meas)
            if not mask.any():
                raise DataFormatError(f"{path}: case `{cid}`: no valid measured point left after filtering.")

            good_rows = np.all(np.isfinite(pred[:, mask]), axis=1)
            cases[cid] = CaseData(cid, tim[mask], meas[mask], pred[good_rows][:, mask],
                                  eval_ids[good_rows])
    return InputData(inputs=inputs, cases=cases)


def write_input_data(path: str | Path, inputs: pd.DataFrame, cases: dict) -> None:
    """Write input_data.h5.

    ``inputs``: DataFrame with an ``eval_id`` column (or index) and one
    column per parameter. ``cases``: case_id -> dict with ``tim``,
    ``meas``, ``pred``, ``eval_ids`` and optionally ``filter``.
    """
    df = inputs.reset_index() if "eval_id" not in inputs.columns else inputs.copy()
    df["eval_id"] = df["eval_id"].astype(np.int64)
    records = df.to_records(index=False)
    with h5py.File(path, "w") as f:
        f.create_dataset("inputs", data=records)
        g_cases = f.create_group("cases")
        for cid, c in cases.items():
            g = g_cases.create_group(str(cid))
            meas = np.asarray(c["meas"], dtype=float).ravel()
            g.create_dataset("tim", data=np.asarray(c.get("tim", np.zeros(meas.shape)), dtype=float).ravel())
            g.create_dataset("meas", data=meas)
            pred = np.asarray(c["pred"], dtype=float)
            if pred.ndim == 1 and meas.shape[0] == 1:
                pred = pred[:, None]
            if pred.ndim != 2 or pred.shape[1] != meas.shape[0]:
                raise DataFormatError(f"case `{cid}`: `pred` has shape {pred.shape}, expected "
                                      f"(n_eval, {meas.shape[0]}).")
            g.create_dataset("pred", data=pred)
            g.create_dataset("eval_ids", data=np.asarray(c["eval_ids"], dtype=np.int64).ravel())
            if c.get("filter") is not None:
                g.create_dataset("filter", data=np.asarray(c["filter"], dtype=bool).ravel())


# --------------------------------------------------------------------------- #
#  synthetic_truth.h5 and validation_<method>.h5
# --------------------------------------------------------------------------- #
def load_synthetic_truth(path: str | Path, n_params: int) -> dict:
    """Return {'population': (p,) or None, 'cases': {case_id: (p,)}}."""
    out = {"population": None, "cases": {}}
    with h5py.File(path, "r") as f:
        if "population" in f:
            out["population"] = np.asarray(f["population"][()], dtype=float).ravel()
            if out["population"].shape[0] != n_params:
                raise DataFormatError(f"{path}: `population` must have {n_params} values.")
        for cid, g in f.get("cases", {}).items():
            theta = np.asarray(g["theta"][()], dtype=float).ravel()
            if theta.shape[0] != n_params:
                raise DataFormatError(f"{path}: case `{cid}`: `theta` must have {n_params} values.")
            out["cases"][cid] = theta
    return out


def load_validation(path: str | Path) -> dict:
    """Return {case_id: {'tim', 'meas', 'pred'}} from validation_<method>.h5."""
    out = {}
    with h5py.File(path, "r") as f:
        if "cases" not in f:
            raise DataFormatError(f"{path}: missing group `cases`.")
        for cid, g in f["cases"].items():
            meas = np.asarray(g["meas"][()], dtype=float).ravel()
            pred = np.atleast_2d(np.asarray(g["pred"][()], dtype=float))
            if meas.shape[0] == 1 and pred.shape[0] == 1:
                pred = pred.T
            if pred.shape[1] != meas.shape[0]:
                raise DataFormatError(f"{path}: case `{cid}`: `pred` must have {meas.shape[0]} columns.")
            tim = np.asarray(g["tim"][()], dtype=float).ravel() if "tim" in g else np.arange(meas.shape[0])
            out[cid] = {"tim": tim, "meas": meas, "pred": pred}
    return out


# --------------------------------------------------------------------------- #
#  Results (HDF5): nested dict <-> groups / datasets / attributes
# --------------------------------------------------------------------------- #
def _write(group: h5py.Group, data: dict) -> None:
    for key, value in data.items():
        key = str(key)
        if isinstance(value, dict):
            _write(group.create_group(key), value)
        elif isinstance(value, np.ndarray) and value.dtype.kind in "fiub":
            group.create_dataset(key, data=value, compression="gzip" if value.size > 1000 else None)
        elif isinstance(value, (list, tuple)) and value and all(isinstance(v, str) for v in value):
            group.attrs[key] = np.array(value, dtype=h5py.string_dtype())
        elif isinstance(value, (list, tuple)):
            group.create_dataset(key, data=np.asarray(value))
        elif value is None:
            group.attrs[key] = "None"
        else:
            group.attrs[key] = value


def save_results(path: str | Path, data: dict) -> Path:
    """Save a nested dict to HDF5. Refuses to overwrite an existing file."""
    path = Path(path)
    if path.exists():
        stem, k = path.stem, 1
        while path.with_name(f"{stem}_{k}{path.suffix}").exists():
            k += 1
        path = path.with_name(f"{stem}_{k}{path.suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        _write(f, data)
    return path


def _read(group: h5py.Group) -> dict:
    out = {}
    for key, value in group.attrs.items():
        if isinstance(value, bytes):
            value = value.decode()
        elif isinstance(value, np.ndarray) and value.dtype.kind in "OS":
            value = [v.decode() if isinstance(v, bytes) else str(v) for v in value]
        out[key] = None if isinstance(value, str) and value == "None" else value
    for key, item in group.items():
        out[key] = _read(item) if isinstance(item, h5py.Group) else item[()]
    return out


def load_results(path: str | Path) -> dict:
    with h5py.File(path, "r") as f:
        return _read(f)

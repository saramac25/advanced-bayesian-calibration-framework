"""Interface between the framework and the user's simulation code.

The framework never assumes a specific solver. To let it run the model
(Morris screening, generation of training data), implement the three
methods of ``ModelInterface`` for your code:

    class MySolver(ModelInterface):
        def write_input(self, theta, case_id, run_dir):
            # write the solver input file(s) in run_dir with the parameter
            # values theta (dict name -> value, physical units)
        def execute(self, case_id, run_dir):
            subprocess.run(["my_solver", "input.txt"], cwd=run_dir, check=True)
        def read_output(self, case_id, run_dir):
            # return the QoI: a float (scalar QoI) or a 1D array (time-dependent)

For a model written in Python, ``PythonModel`` wraps a function directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class ModelInterface(ABC):
    """Abstract solver interface."""

    @abstractmethod
    def write_input(self, theta: dict, case_id: str, run_dir: Path) -> None:
        ...

    @abstractmethod
    def execute(self, case_id: str, run_dir: Path) -> None:
        ...

    @abstractmethod
    def read_output(self, case_id: str, run_dir: Path) -> np.ndarray:
        ...

    def run(self, theta: dict, case_id: str, run_dir: str | Path) -> np.ndarray:
        """Write the input, run the solver and return the QoI as a 1D array."""
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        self.write_input(theta, case_id, run_dir)
        self.execute(case_id, run_dir)
        return np.atleast_1d(np.asarray(self.read_output(case_id, run_dir), dtype=float))


class PythonModel(ModelInterface):
    """Wrap a Python function ``f(theta: dict, case_id: str) -> QoI``. No files are written."""

    def __init__(self, func):
        self.func = func
        self._last = None

    def write_input(self, theta, case_id, run_dir):
        self._theta = theta

    def execute(self, case_id, run_dir):
        self._last = self.func(self._theta, case_id)

    def read_output(self, case_id, run_dir):
        return self._last

    def run(self, theta, case_id, run_dir=None):
        return np.atleast_1d(np.asarray(self.func(theta, case_id), dtype=float))

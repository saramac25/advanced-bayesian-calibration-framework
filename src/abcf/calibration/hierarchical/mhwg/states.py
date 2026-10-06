"""State of the hierarchical sampler.

Original author: Gustav Robertson (created 20 Sep 2024).
Changes for abcf: none (docstrings only).
"""


class state:
    """mu, sig: (p,) population mean and std; X: (n_cases, p) case-level values."""

    def __init__(self, mu, sig, X):
        self.mu = mu
        self.sig = sig
        self.X = X

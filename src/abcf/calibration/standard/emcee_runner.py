"""Standard (non-hierarchical) Bayesian calibration with emcee.

One set of calibrated parameters shared by all cases, uniform prior over
the training-data range (standardised units), Gaussian likelihood of
``abcf.calibration.likelihood``. Sampler: affine-invariant ensemble
sampler (Goodman & Weare, 2010; Foreman-Mackey et al., 2013) with the
stretch move; its scale ``a`` is set in the configuration.

Tuning in two steps, as in the thesis scripts:
  1. number of walkers: run ``nsteps_tuning`` steps for each multiplier
     and keep the fastest configuration that meets the R-hat and
     acceptance criteria;
  2. number of steps: with that number of walkers, run each value of
     ``nsteps_list`` and keep the fastest that also meets the ESS
     criterion.
"""

from __future__ import annotations

import time

import emcee
import numpy as np

from ...diagnostics.advice import standard_advice
from ...diagnostics.convergence import chain_diagnostics


class StandardPosterior:
    """log p(x | data) up to a constant, x = standardised calibrated parameters."""

    def __init__(self, likelihoods: list, lower: np.ndarray, upper: np.ndarray):
        self.likelihoods = likelihoods
        self.lower, self.upper = lower, upper

    def __call__(self, x: np.ndarray) -> float:
        if np.any(x < self.lower) or np.any(x > self.upper):
            return -np.inf
        return float(sum(lik(x) for lik in self.likelihoods))


def run_emcee(log_prob, lower, upper, nwalkers: int, nsteps: int, stretch_a: float,
              burnin_fraction: float, seed: int, progress: bool = True) -> dict:
    """One emcee run. Chains are returned with shape (nwalkers, nsteps, n_dim)."""
    ndim = len(lower)
    rng = np.random.default_rng(seed)
    p0 = rng.uniform(lower, upper, size=(nwalkers, ndim))
    np.random.seed(seed)        # emcee draws from the global NumPy generator
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob,
                                    moves=emcee.moves.StretchMove(a=stretch_a))
    t0 = time.time()
    sampler.run_mcmc(p0, nsteps, progress=progress)
    wall = time.time() - t0
    chains = np.transpose(sampler.get_chain(), (1, 0, 2))          # (walkers, steps, dim)
    logp = sampler.get_log_prob().T                                 # (walkers, steps)
    burnin = int(round(burnin_fraction * nsteps))
    return {"chains": chains, "log_prob": logp, "burnin": burnin, "nwalkers": nwalkers,
            "nsteps": nsteps, "acceptance": sampler.acceptance_fraction, "wall_clock_s": wall,
            "stretch_a": stretch_a, "seed": seed}


def _passes(run: dict, cfg: dict, use_ess: bool) -> tuple:
    d = chain_diagnostics(run["chains"][:, run["burnin"]:, :])
    acc = float(run["acceptance"].mean())
    advice = standard_advice(acc, float(np.max(d["rhat"])),
                             float(np.min(d["ess_bulk"])) if use_ess else np.inf,
                             run["nwalkers"], run["nsteps"], cfg)
    return not advice, d, acc, advice


def tune_and_run(log_prob, lower, upper, cfg: dict, n_dim: int, progress: bool = True):
    """Two-step tuning; returns (final run dict, tuning table rows)."""
    s, dg = cfg["standard"], cfg["diagnostics"]
    seed = int(cfg["project"]["seed"])
    rows = []

    def record(step, run, ok, d, acc, advice):
        rows.append({"step": step, "nwalkers": run["nwalkers"], "nsteps": run["nsteps"],
                     "rhat_max": float(np.max(d["rhat"])), "ess_bulk_min": float(np.min(d["ess_bulk"])),
                     "ess_threshold": dg["ess_per_chain"] * run["nwalkers"],
                     "acceptance_mean": float(acc), "wall_clock_s": run["wall_clock_s"], "pass": ok,
                     "suggestion": " | ".join(advice)})

    # Step 1: number of walkers
    mults = list(s["walker_multipliers"])
    if len(mults) == 1:
        nwalkers = max(int(mults[0] * n_dim), 2 * n_dim + 2)
    else:
        cand = []
        for m in mults:
            nw = max(int(m * n_dim), 2 * n_dim + 2)
            run = run_emcee(log_prob, lower, upper, nw, s["nsteps_tuning"], s["stretch_a"],
                            s["burnin_fraction"], seed + m, progress)
            ok, d, acc, advice = _passes(run, cfg, use_ess=False)
            record("1_walkers", run, ok, d, acc, advice)
            cand.append((ok, run["wall_clock_s"], np.max(d["rhat"]), nw))
        good = [c for c in cand if c[0]]
        if good:
            nwalkers = min(good, key=lambda c: c[1])[3]
        else:
            nwalkers = min(cand, key=lambda c: c[2])[3]
            print("\nStandard calibration, step 1: no walker number meets the criteria; "
                  f"using the lowest R-hat ({nwalkers} walkers). See the 'suggestion' column "
                  "of the tuning table.")

    # Step 2: number of steps
    runs = []
    for n in s["nsteps_list"]:
        run = run_emcee(log_prob, lower, upper, nwalkers, n, s["stretch_a"],
                        s["burnin_fraction"], seed + 1000 + n, progress)
        ok, d, acc, advice = _passes(run, cfg, use_ess=True)
        record("2_steps", run, ok, d, acc, advice)
        runs.append((ok, run, advice))
    good = [r for ok, r, _ in runs if ok]
    if good:
        final = min(good, key=lambda r: r["wall_clock_s"])
    else:
        final, advice = runs[-1][1], runs[-1][2]
        print("\nStandard calibration: no run meets all criteria; the longest run is kept.")
        for a in advice:
            print(f"  SUGGESTION: {a}")
    return final, rows

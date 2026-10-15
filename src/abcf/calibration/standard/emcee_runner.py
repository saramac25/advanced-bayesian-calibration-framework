"""Standard (non-hierarchical) Bayesian calibration with emcee.

One set of calibrated parameters shared by all cases, uniform prior over
the training-data range (standardised units) or a normal prior truncated to
that range (``parameters.<name>.prior``), Gaussian likelihood of
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
from ...diagnostics.criteria import standard_rows


class StandardPosterior:
    """log p(x | data) up to a constant, x = standardised calibrated parameters."""

    def __init__(self, likelihoods: list, lower: np.ndarray, upper: np.ndarray, prior=None):
        self.likelihoods = likelihoods
        self.lower, self.upper = lower, upper
        self.prior = prior          # (mean_s, std_s) of normal priors, NaN = uniform; or None

    def __call__(self, x: np.ndarray) -> float:
        if np.any(x < self.lower) or np.any(x > self.upper):
            return -np.inf
        from ...config import normal_logprior
        return normal_logprior(x, self.prior) + float(sum(lik(x) for lik in self.likelihoods))


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


class ContinuedEmcee:
    """One emcee ensemble extended to increasing lengths (``standard.continue_chains``):
    ``extend(n)`` runs only the steps missing to reach n in total and returns the run dict
    of the whole chain (burn-in = burnin_fraction x total length). The stretch move has
    no adaptation, so a continued chain is identical in law to a chain run at once."""

    def __init__(self, log_prob, lower, upper, nwalkers: int, stretch_a: float,
                 burnin_fraction: float, seed: int):
        ndim = len(lower)
        rng = np.random.default_rng(seed)
        self.p0 = rng.uniform(lower, upper, size=(nwalkers, ndim))
        np.random.seed(seed)
        self.sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob,
                                             moves=emcee.moves.StretchMove(a=stretch_a))
        self.nwalkers, self.stretch_a, self.f, self.seed = nwalkers, stretch_a, burnin_fraction, seed
        self.done, self.wall = 0, 0.0

    def extend(self, nsteps: int, progress: bool = False) -> dict:
        n_new = nsteps - self.done
        if n_new > 0:
            t0 = time.time()
            self.sampler.run_mcmc(self.p0 if self.done == 0 else None, n_new, progress=progress)
            self.wall += time.time() - t0
            self.done = nsteps
        chains = np.transpose(self.sampler.get_chain(), (1, 0, 2))
        return {"chains": chains, "log_prob": self.sampler.get_log_prob().T,
                "burnin": int(round(self.f * nsteps)), "nwalkers": self.nwalkers, "nsteps": nsteps,
                "acceptance": self.sampler.acceptance_fraction, "wall_clock_s": self.wall,
                "stretch_a": self.stretch_a, "seed": self.seed}


def _passes(run: dict, cfg: dict, use_ess: bool) -> tuple:
    """(ok, rows, acceptance, advice). ``use_ess=False`` (walker tuning): only R-hat and
    acceptance. With ``use_ess``, every active criterion of ``diagnostics``."""
    names = [f"x{j}" for j in range(run["chains"].shape[2])]
    rows = standard_rows(run, cfg, names)
    acc = float(run["acceptance"].mean())
    rhat = max(r["rhat"] for r in rows) if cfg["diagnostics"].get("standard_rhat", True) else 0.0
    ess = min(r["ess_bulk"] for r in rows) if use_ess else np.inf
    advice = standard_advice(acc, rhat, ess, run["nwalkers"], run["nsteps"], cfg)
    if use_ess:
        tau_bad = [r for r in rows if not r["tau_ok"]]
        if tau_bad:
            f = cfg["diagnostics"]["tau_factor"]
            advice.append(f"N = {run['nsteps'] - run['burnin']} post-burn-in steps < {f} tau "
                          f"(tau = {max(r['tau'] for r in tau_bad):.1f}): run longer.")
        conv = all(r["converged"] for r in rows)
        ok = conv and all(r["acceptance_ok"] for r in rows)
    else:
        conv = ok = not advice
    return ok, rows, acc, advice, conv


def tune_and_run(log_prob, lower, upper, cfg: dict, n_dim: int, progress: bool = True):
    """Two-step tuning; returns (final run dict, tuning table rows)."""
    s, dg = cfg["standard"], cfg["diagnostics"]
    seed = int(cfg["project"]["seed"])
    rows = []

    def record(step, run, ok, d, acc, advice, conv=None):
        rows.append({"step": step, "nwalkers": run["nwalkers"], "nsteps": run["nsteps"],
                     "rhat_max": max(r["rhat"] for r in d), "ess_min": min(r["ess_bulk"] for r in d),
                     "ess_threshold": dg["ess_per_chain"] * run["nwalkers"],
                     "tau_max": max(r["tau"] for r in d), "n_over_tau_min": min(r["n_over_tau"] for r in d),
                     "acceptance_mean": float(acc), "wall_clock_s": run["wall_clock_s"],
                     "converged": conv if conv is not None else ok, "pass": ok,
                     "suggestion": " | ".join(advice)})

    # emcee needs at least 2 x n_dim walkers
    min_walkers = 2 * n_dim

    # Step 1: number of walkers
    mults = list(s["walker_multipliers"])
    if len(mults) == 1:
        nwalkers = max(int(mults[0] * n_dim), min_walkers)
    else:
        cand = []
        for m in mults:
            nw = max(int(m * n_dim), min_walkers)
            run = run_emcee(log_prob, lower, upper, nw, s["nsteps_tuning"], s["stretch_a"],
                            s["burnin_fraction"], seed + m, progress)
            ok, d, acc, advice, _ = _passes(run, cfg, use_ess=False)
            record("1_walkers", run, ok, d, acc, advice)
            cand.append((ok, run["wall_clock_s"], max(r["rhat"] for r in d), nw))
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
    lengths = sorted(s["nsteps_list"]) if s.get("stop_at_first_pass") else list(s["nsteps_list"])
    cont = bool(s.get("continue_chains", True)) and s.get("stop_at_first_pass")
    ens = ContinuedEmcee(log_prob, lower, upper, nwalkers, s["stretch_a"], s["burnin_fraction"],
                         seed + 1000 + lengths[0]) if cont else None
    if cont and len(lengths) > 1:
        print("  longer lengths continue the previous walkers (standard.continue_chains)")
    for n in lengths:
        if cont:
            run = ens.extend(n, progress)
        else:
            run = run_emcee(log_prob, lower, upper, nwalkers, n, s["stretch_a"],
                            s["burnin_fraction"], seed + 1000 + n, progress)
        ok, d, acc, advice, conv = _passes(run, cfg, use_ess=True)
        record("2_steps", run, ok, d, acc, advice, conv)
        runs.append((ok, run, advice, conv))
        if progress or s.get("stop_at_first_pass"):
            print(f"  emcee {nwalkers} walkers x {n} steps: converged = {conv}, "
                  f"acceptance = {acc:.3f}, pass = {ok}")
        if s.get("stop_at_first_pass") and conv:
            break       # length found; the acceptance is checked at this length
    if s.get("stop_at_first_pass"):
        final, advice = runs[-1][1], runs[-1][2]
        if not runs[-1][3]:
            print("\nStandard calibration: the longest length does not meet the convergence criteria.")
        if advice:
            for a in advice:
                print(f"  SUGGESTION: {a}")
        return final, rows
    good = [r for ok, r, _, _ in runs if ok]
    if good:
        final = min(good, key=lambda r: r["wall_clock_s"])
    else:
        final, advice = runs[-1][1], runs[-1][2]
        print("\nStandard calibration: no run meets all criteria; the longest run is kept.")
        for a in advice:
            print(f"  SUGGESTION: {a}")
    return final, rows

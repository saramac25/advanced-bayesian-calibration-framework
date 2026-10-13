"""Convergence and tuning criteria, shared by steps 04, 05 and 06.

For every sampled quantity (a parameter of the standard calibration, or
a block mu_d, sigma_d, X[case, d] of the hierarchical one) the table
reports:

* R-hat (rank-normalised, split) and bulk/tail ESS, Vehtari et al. (2021),
  doi:10.1214/20-BA1221;
* the integrated autocorrelation time tau and the check N >= tau_factor * tau,
  N = post-burn-in draws per chain (Foreman-Mackey et al., 2013,
  doi:10.1086/670067). For emcee, tau is emcee's estimate (autocorrelation
  averaged over the walkers); otherwise tau = N_total / bulk-ESS;
* MCSE/sigma (overlapping batch means), checked only if a threshold is set;
* the acceptance rate and its range.

Two verdicts: ``converged`` (R-hat, ESS, tail-ESS, tau, MCSE: decides the
length of the chains) and ``pass`` (converged AND acceptance in range).
Which criteria are active is set in the ``diagnostics`` block of the
configuration.
"""

from __future__ import annotations

import warnings

import numpy as np

from .convergence import chain_diagnostics


def emcee_tau(chains: np.ndarray) -> np.ndarray:
    """emcee integrated autocorrelation time per parameter.

    chains: (n_walkers, n_draws, p) post burn-in. Uses
    ``emcee.autocorr.integrated_time`` (autocorrelation averaged over the
    walkers, Sokal's automatic window), without its length warning: the
    N >= tau_factor * tau check is done here.
    """
    import logging

    import emcee
    logging.getLogger("emcee").setLevel(logging.ERROR)
    x = np.transpose(np.asarray(chains, dtype=float), (1, 0, 2))     # (n_draws, n_walkers, p)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.atleast_1d(emcee.autocorr.integrated_time(x, quiet=True))


def criteria_rows(chains: np.ndarray, names: list, cfg: dict, method: str, block: str,
                  case: str, acceptance, acc_range, convergence_required: bool = True) -> list:
    """One row per parameter of a block.

    chains: (n_chains, n_draws, p) post burn-in; acceptance: scalar or (p,).
    ``convergence_required=False``: the block is reported, but only its
    acceptance enters the verdict (e.g. X blocks when only mu and sigma
    must converge).
    """
    chains = np.asarray(chains, dtype=float)
    if chains.ndim == 2:
        chains = chains[:, :, None]
    dg = cfg["diagnostics"]
    n_chains, n_draws = chains.shape[:2]
    d = chain_diagnostics(chains)
    sc_autocorr = method == "SC" and dg.get("standard_ess", "arviz") == "autocorr"
    if sc_autocorr:
        tau = emcee_tau(chains)
        ess = n_draws * n_chains / tau
    else:
        ess = d["ess_bulk"]
        tau = n_draws * n_chains / np.maximum(ess, 1e-12)
    use_rhat = not (method == "SC" and not dg.get("standard_rhat", True))
    use_tail = bool(dg.get("check_tail_ess")) and method != "SC"
    thr = dg["ess_per_chain"] * n_chains
    tau_factor = dg.get("tau_factor")

    rows = []
    for j, n in enumerate(names):
        acc = acceptance[j] if np.ndim(acceptance) else acceptance
        acc = float(acc) if acc is not None else np.nan
        lo, hi = acc_range
        r = {"block": block, "case": case, "parameter": n,
             "rhat": float(d["rhat"][j]), "ess_bulk": float(ess[j]), "ess_tail": float(d["ess_tail"][j]),
             "ess_threshold": thr, "tau": float(tau[j]), "n_per_chain": n_draws,
             "n_over_tau": float(n_draws / tau[j]) if tau[j] > 0 else np.inf,
             "mean": float(d["mean"][j]), "std": float(d["std"][j]),
             "mcse": float(d["mcse"][j]), "mcse_over_std": float(d["mcse_over_std"][j]),
             "acceptance": acc, "acceptance_range": (lo, hi)}
        r["rhat_ok"] = bool(r["rhat"] <= dg["rhat_max"]) if use_rhat else True
        r["ess_ok"] = bool(r["ess_bulk"] >= thr)
        r["ess_tail_ok"] = bool(r["ess_tail"] >= thr) if use_tail else True
        r["tau_ok"] = bool(r["n_over_tau"] >= tau_factor) if tau_factor else True
        r["mcse_ok"] = bool(r["mcse_over_std"] <= dg["mcse_over_std_max"]) \
            if dg.get("mcse_over_std_max") is not None else True
        r["acceptance_ok"] = bool(lo <= acc <= hi) if acc == acc else True
        conv = r["rhat_ok"] and r["ess_ok"] and r["ess_tail_ok"] and r["tau_ok"] and r["mcse_ok"]
        r["convergence_required"] = convergence_required
        r["converged"] = bool(conv) if convergence_required else True
        r["pass"] = bool(r["converged"] and r["acceptance_ok"])
        rows.append(r)
    return rows


def standard_rows(run_or_results, cfg: dict, names: list) -> list:
    """Rows for an emcee run (dict from run_emcee) or a loaded SC results file."""
    if "chains" in run_or_results and isinstance(run_or_results["chains"], dict):
        ch, burnin = run_or_results["chains"], int(run_or_results["metadata"]["burnin"])
        theta, acc = ch["theta"], float(np.mean(ch["acceptance"]))
    else:
        theta, burnin = run_or_results["chains"], int(run_or_results["burnin"])
        acc = float(np.mean(run_or_results["acceptance"]))
    return criteria_rows(theta[:, burnin:, :], names, cfg, "SC", "theta", "", acc,
                         cfg["diagnostics"]["acceptance_range_standard"])


def hierarchical_rows(ch: dict, burnin: int, names: list, cases: list, cfg: dict) -> list:
    """Rows for every block of a hierarchical run (chains dict as stored in HC_*.h5)."""
    dg = cfg["diagnostics"]
    mu, sig, X = ch["mu"], ch["sigma"], ch["X"]
    p = mu.shape[2]
    range_1d = dg["acceptance_range_1d"]
    range_X = range_1d if p == 1 else dg["acceptance_range_X"]
    required = set(dg.get("hierarchical_convergence_blocks") or ["mu", "sigma", "X"])
    acc_mu = ch["acceptance_mu"][:, burnin:, :].mean(axis=(0, 1))
    acc_sig = ch["acceptance_sigma"][:, burnin:, :].mean(axis=(0, 1))
    acc_X = ch["acceptance_X"][:, burnin:, :].mean(axis=(0, 1))
    rows = criteria_rows(mu[:, burnin:], names, cfg, "HC", "mu", "", acc_mu, range_1d, "mu" in required)
    rows += criteria_rows(sig[:, burnin:], names, cfg, "HC", "sigma", "", acc_sig, range_1d,
                          "sigma" in required)
    for e, cid in enumerate(cases):
        rows += criteria_rows(X[:, burnin:, e, :], names, cfg, "HC", "X", cid, acc_X[e], range_X,
                              "X" in required)
    return rows

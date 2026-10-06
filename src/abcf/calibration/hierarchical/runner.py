"""Hierarchical Bayesian calibration: MH-within-Gibbs sampler.

Model (standardised calibrated parameters, d = 1..p, cases e = 1..E):

    X[e, d] ~ N(mu_d, sigma_d^2)                     inter-case variability
    z_meas[e] ~ N(z_GP(X[e]), Sigma_meas[e] [+ var_GP])   likelihood
    mu_d ~ U(training range),  sigma_d ~ U(0, sigma_upper_bound]

Each Gibbs sweep updates mu_d and sigma_d (one block per parameter) and
then X[e] (one block per case) with Metropolis-Hastings steps, using
the sampler building blocks of Gustav Robertson (``mhwg`` sub-package;
Robertson et al., 2025). Optional joint shift move (Sara Maccario): after
each sweep, mu_d and X[:, d] are shifted by the same amount, which
leaves the hierarchical term unchanged and lets mu move when sigma is
small.

The state holds ONLY the calibrated parameters; the fixed ones are
added by ``ParameterExpander`` before every GP call.
"""

from __future__ import annotations

import time

import numpy as np
from tqdm import tqdm


def _import_mhwg():
    try:
        from .mhwg import adaptation, custom_logps, proposals, random_walk, states
    except ImportError as err:      # pragma: no cover - message for the user
        raise ImportError(
            "The MH-within-Gibbs building blocks are missing from "
            "src/abcf/calibration/hierarchical/mhwg/ (see the README in that folder). "
            f"Original error: {err}") from err
    return adaptation, custom_logps, proposals, random_walk, states


def _initial_state(rng, chain: int, h: dict, lower, upper, n_cases: int):
    p = len(lower)
    if chain == 0 or not h["dispersed_init"]:
        mu = 0.5 * (lower + upper)
        sig = np.full(p, float(h["initial_sigma"]))
    else:
        mu = rng.uniform(lower, upper)
        sig = rng.uniform(0.2, 1.0, size=p) * float(h["initial_sigma"]) * 2
    X = np.clip(mu + sig * rng.standard_normal((n_cases, p)), lower, upper)
    return mu, sig, X


def _build_samplers(mh, rng, h: dict, case_ids: list, likelihoods: list, lower, upper, nsteps):
    adaptation, custom_logps, proposals, random_walk, _ = mh
    p = len(lower)
    burnin = int(round(h["burnin_fraction"] * nsteps))
    ad = h["adaptation"]
    v0 = h["initial_proposal_variance"]

    def adapt(kind, index=None):
        if ad["type"] == "acceptance_rate":
            return adaptation.adapt_acceptance_rate(burnin=burnin, adapt_start=ad["start"])
        common = dict(adapt_start=ad["start"], proposal_update_start=ad["proposal_update_start"],
                      adapt_stop=burnin)
        if kind == "mu":
            return adaptation.adapt_covariance_matrix_mu_d(index, **common)
        if kind == "sig":
            return adaptation.adapt_covariance_matrix_sig_d(index, **common)
        return adaptation.adapt_covariance_matrix_X(index, epsilon=ad["epsilon"], s_d=ad["s_d"], **common)

    hyper = []
    for d in range(p):
        hyper.append(random_walk.mh_sampler(
            proposals.propose_mu_d(np.eye(1) * v0["mu"], d, rng=rng),
            custom_logps.logp_mu(lb=lower, ub=upper),
            recompute_logp=True, rng=rng, adaptation=adapt("mu", d)))
        hyper.append(random_walk.mh_sampler(
            proposals.propose_sig_d(np.eye(1) * v0["sigma"], d, rng=rng),
            custom_logps.logp_sig(lb=0.0, ub=float(h["sigma_upper_bound"])),
            recompute_logp=True, rng=rng, adaptation=adapt("sig", d)))

    fixed_cov = h.get("fixed_proposal_cov") or {}
    x_samplers = []
    for e, (cid, lik) in enumerate(zip(case_ids, likelihoods)):
        logp = custom_logps.logp_X(e, lik, lik.z_meas, lik.meas_cov, lb=lower, ub=upper)
        if cid in fixed_cov:
            cov = np.asarray(fixed_cov[cid], dtype=float)
            if cov.shape != (p, p):
                raise ValueError(f"hierarchical.fixed_proposal_cov.{cid} must be {p}x{p} "
                                 f"(one row/column per calibrated parameter).")
            prop = proposals.propose_X(cov, e, rng=rng)
            # The value in the config IS the proposal covariance (no 0.1 factor, no adaptation)
            prop.set_adaptation_factor(1.0)
            x_samplers.append(random_walk.mh_sampler(prop, logp, recompute_logp=True, rng=rng))
        else:
            cov = np.eye(p) * v0["X"]
            x_samplers.append(random_walk.mh_sampler(
                proposals.propose_X(cov, e, rng=rng), logp, recompute_logp=True, rng=rng,
                adaptation=adapt("X", e)))
    return hyper, x_samplers


def run_chain(chain: int, likelihoods: list, case_ids: list, lower, upper, cfg: dict,
              progress: bool = True) -> dict:
    """Run one chain; returns histories in standardised units."""
    mh = _import_mhwg()
    _, custom_logps, _, _, states = mh
    h = cfg["hierarchical"]
    nsteps = int(h["nsteps"])
    burnin = int(round(h["burnin_fraction"] * nsteps))
    seed = int(cfg["project"]["seed"]) + 1000 * chain
    rng = np.random.default_rng(seed)
    p, E = len(lower), len(case_ids)

    mu, sig, X = _initial_state(rng, chain, h, lower, upper, E)
    st = states.state(mu, sig, X)
    hyper, x_samplers = _build_samplers(mh, rng, h, case_ids, likelihoods, lower, upper, nsteps)
    logp_hyper = custom_logps.logp_hyper()

    shift = h["joint_shift_move"]
    shift_std = np.full(p, float(shift["initial_std"]))

    mu_h = np.empty((nsteps, p))
    sig_h = np.empty((nsteps, p))
    X_h = np.empty((nsteps, E, p))
    logpost = np.full(nsteps, np.nan)
    acc_shift = np.zeros((nsteps, p), dtype=bool)

    def total_loglik(Xc):
        return sum(lik(Xc[e]) for e, lik in enumerate(likelihoods))

    t0 = time.time()
    for i in tqdm(range(nsteps), desc=f"HC chain {chain + 1}", disable=not progress):
        for smp in hyper:
            st = smp.step(st)
        for smp in x_samplers:
            st = smp.step(st)

        if shift["enabled"]:
            ll_old = total_loglik(st.X)
            for d in range(p):
                delta = rng.normal(0.0, shift_std[d])
                mu_new, X_new = st.mu.copy(), st.X.copy()
                mu_new[d] += delta
                X_new[:, d] += delta
                inside = (lower[d] <= mu_new[d] <= upper[d]) and \
                    np.all((X_new[:, d] >= lower[d]) & (X_new[:, d] <= upper[d]))
                if not inside:
                    continue
                # The hierarchical term is invariant under the shift and the
                # priors are uniform: the acceptance depends on the likelihood only.
                ll_new = total_loglik(X_new)
                if np.log(rng.uniform()) < ll_new - ll_old:
                    st = states.state(mu_new, st.sig.copy(), X_new)
                    ll_old = ll_new
                    acc_shift[i, d] = True
            every = int(shift["adapt_every"])
            if i < burnin and (i + 1) % every == 0:
                rate = acc_shift[i + 1 - every:i + 1].mean(axis=0)
                shift_std *= np.exp(rate - float(shift["target_acceptance"]))

        mu_h[i], sig_h[i], X_h[i] = st.mu, st.sig, st.X
        if h["store_logpost"]:
            logpost[i] = logp_hyper(st) + total_loglik(st.X)

    return {"mu": mu_h, "sigma": sig_h, "X": X_h, "logpost": logpost,
            "acceptance_shift": acc_shift, "shift_std_final": shift_std,
            "wall_clock_s": time.time() - t0, "seed": seed, "burnin": burnin, "nsteps": nsteps}


def run_hierarchical(likelihoods: list, case_ids: list, lower, upper, cfg: dict,
                     progress: bool = True) -> dict:
    """Run ``n_chains`` independent chains (different seeds; dispersed starts if requested)."""
    chains = [run_chain(c, likelihoods, case_ids, lower, upper, cfg, progress)
              for c in range(int(cfg["hierarchical"]["n_chains"]))]
    stack = lambda k: np.stack([ch[k] for ch in chains])   # noqa: E731
    return {"mu": stack("mu"), "sigma": stack("sigma"), "X": stack("X"),
            "logpost": stack("logpost"), "acceptance_shift": stack("acceptance_shift"),
            "shift_std_final": stack("shift_std_final"),
            "wall_clock_s": np.array([ch["wall_clock_s"] for ch in chains]),
            "seeds": np.array([ch["seed"] for ch in chains]),
            "burnin": chains[0]["burnin"], "nsteps": chains[0]["nsteps"]}

"""Hierarchical Bayesian calibration: MH-within-Gibbs sampler.

Model (standardised calibrated parameters, d = 1..p, cases e = 1..E):

    X[e, d] ~ N(mu_d, sigma_d^2)                     inter-case variability
    z_meas[e] ~ N(z_GP(X[e]), Sigma_meas[e] [+ var_GP])   likelihood
    mu_d ~ U(training range) or N(m_d, s_d^2) truncated to it (parameters.<name>.prior),
    sigma_d ~ U(0, sigma_upper_bound]

Each Gibbs sweep updates mu_d and sigma_d (one block per parameter) and
then X[e] (one block per case) with Metropolis-Hastings steps, using
the sampler building blocks of Gustav Robertson (``mhwg`` sub-package;
Robertson et al., 2025). Optional joint shift move (Sara Maccario): after
each sweep, mu_d and X[:, d] are shifted by the same amount, which
leaves the hierarchical term unchanged and lets mu move when sigma is
small. Optional joint scale move: sigma_d and the deviations X[:, d] - mu_d
are multiplied by the same factor, which lets sigma and X move together when
sigma is small (funnel of hierarchical models).

The state holds ONLY the calibrated parameters; the fixed ones are
added by ``ParameterExpander`` before every GP call.
"""

from __future__ import annotations

import os
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
    """Starting point of a chain. Default: X[e] drawn around mu. With
    ``init_X: case_mode`` the case-level parameters start at the maximum of the likelihood
    of their own case (computed once in run_hierarchical) plus a small random jitter, while
    mu and sigma are still dispersed between the chains: this avoids chains whose X[e]
    never reach a very narrow likelihood peak, without tying the chains to one point."""
    p = len(lower)
    modes = h.get("_case_modes")
    if chain == 0 or not h["dispersed_init"]:
        if modes is not None:
            m = np.asarray(modes, float)
            mu, sig = m.mean(axis=0), np.maximum(m.std(axis=0), 0.05)
        else:
            mu = 0.5 * (lower + upper)
            sig = np.full(p, float(h["initial_sigma"]))
    else:
        mu = rng.uniform(lower, upper)
        sig = rng.uniform(0.2, 1.0, size=p) * float(h["initial_sigma"]) * 2
    if modes is not None:
        jit = float(h.get("init_X_jitter", 0.05))
        X = np.clip(np.asarray(modes, float) + jit * rng.standard_normal((n_cases, p)), lower, upper)
    else:
        X = np.clip(mu + sig * rng.standard_normal((n_cases, p)), lower, upper)
    return mu, sig, X


def case_modes(likelihoods: list, lower, upper, n_starts: int = 30, seed: int = 0):
    """Maximum of the likelihood of every case over the prior box (standardised units):
    best of n_starts random points refined with Nelder-Mead. Returns (modes (E, p), loglik (E,))."""
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    modes, vals = [], []
    for lik in likelihoods:
        def f(x):
            if np.any(x < lower) or np.any(x > upper):
                return np.inf
            v = lik(x)
            return -v if np.isfinite(v) else np.inf
        starts = np.vstack([0.5 * (lower + upper), rng.uniform(lower, upper, size=(n_starts, len(lower)))])
        fs = np.array([f(x) for x in starts])
        best = None
        for x0 in starts[np.argsort(fs)[:3]]:
            r = minimize(f, x0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 2000})
            if best is None or r.fun < best.fun:
                best = r
        modes.append(np.clip(best.x, lower, upper))
        vals.append(-best.fun)
    return np.array(modes), np.array(vals)


def _build_samplers(mh, rng, h: dict, case_ids: list, likelihoods: list, lower, upper, nsteps,
                    frozen: list | None = None):
    """``frozen``: list of (cov, adaptation_factor), one per sampler (hyper first, then X),
    from a previous segment of the same chain: the proposals are set to these values and
    not adapted any more (continuation of a chain after its adaptation phase)."""
    adaptation, custom_logps, proposals, random_walk, _ = mh
    p = len(lower)
    burnin = int(round(h["burnin_fraction"] * nsteps))
    ad = h["adaptation"]
    # the adaptation stops `stop_before_burnin_end` iterations before the end of the burn-in,
    # so the last part of the burn-in already runs with the final (fixed) proposal
    adapt_end = burnin - int(ad.get("stop_before_burnin_end") or 0)
    v0 = h["initial_proposal_variance"]

    def adapt(kind, index=None):
        if frozen is not None:
            return None
        block = {"mu": "mu", "sig": "sigma", "X": "X"}[kind]
        if ad[block] == "acceptance_rate":
            return adaptation.adapt_acceptance_rate(
                burnin=adapt_end, adapt_start=ad["start"], adapt_rate=float(ad["rate"]),
                window=ad["window"], target_acceptance_rate=float(ad["target_acceptance"][block]))
        s_d = ad["s_d"].get(block) if isinstance(ad["s_d"], dict) else ad["s_d"]
        common = dict(adapt_start=ad["start"], proposal_update_start=ad["proposal_update_start"],
                      adapt_stop=adapt_end, epsilon=float(ad["epsilon"]),
                      s_d=None if s_d is None else float(s_d))
        if kind == "mu":
            return adaptation.adapt_covariance_matrix_mu_d(index, **common)
        if kind == "sig":
            return adaptation.adapt_covariance_matrix_sig_d(index, **common)
        return adaptation.adapt_covariance_matrix_X(index, **common)

    prior = (h.get("_normal_prior_mu") or None)
    prior = None if prior is None else (np.asarray(prior[0], float), np.asarray(prior[1], float))

    class _LogpMu(custom_logps.logp_mu):
        """logp_mu (hierarchical term, bounds) + normal prior on mu where given."""
        def __call__(self, state):
            v = super().__call__(state)
            if prior is None or not np.isfinite(v):
                return v
            from ...config import normal_logprior
            return v + normal_logprior(state.mu, prior)

    hyper = []
    for d in range(p):
        hyper.append(random_walk.mh_sampler(
            proposals.propose_mu_d(np.eye(1) * v0["mu"], d, rng=rng),
            _LogpMu(lb=lower, ub=upper),
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
    if frozen is not None:
        for smp, (cov, fac) in zip(hyper + x_samplers, frozen):
            smp.proposal.cov = np.array(cov, dtype=float)
            smp.proposal.adaptation_factor = float(fac)
    return hyper, x_samplers


def run_chain(chain: int, likelihoods: list, case_ids: list, lower, upper, cfg: dict,
              progress: bool = True, nsteps: int | None = None, resume: dict | None = None) -> dict:
    """Run one chain; returns histories in standardised units.

    ``resume``: checkpoint returned by a previous call for the same chain. The chain then
    continues from its last state, with the same random-number stream and the proposals
    (block covariances, joint-move step sizes) frozen at the end of the previous adaptation,
    for ``nsteps`` further iterations; the returned histories contain only these iterations.
    Without ``resume`` the chain starts from scratch and adapts during its burn-in."""
    mh = _import_mhwg()
    _, custom_logps, _, _, states = mh
    h = cfg["hierarchical"]
    nsteps = int(nsteps or h["nsteps"])
    burnin = int(round(h["burnin_fraction"] * nsteps))
    seed = int(cfg["project"]["seed"]) + 1000 * chain
    rng = np.random.default_rng(seed)
    p, E = len(lower), len(case_ids)

    if resume is None:
        mu, sig, X = _initial_state(rng, chain, h, lower, upper, E)
        frozen = None
    else:
        rng.bit_generator.state = resume["rng_state"]
        mu, sig, X = (np.array(resume[k], dtype=float) for k in ("mu", "sig", "X"))
        frozen = resume["proposals"]
    st = states.state(mu, sig, X)
    hyper, x_samplers = _build_samplers(mh, rng, h, case_ids, likelihoods, lower, upper, nsteps,
                                        frozen=frozen)
    logp_hyper = custom_logps.logp_hyper()
    from ...config import normal_logprior
    pm = h.get("_normal_prior_mu")
    prior_mu = None if pm is None else (np.asarray(pm[0], float), np.asarray(pm[1], float))

    shift = h["joint_shift_move"]
    adapt_end = burnin - int(h["adaptation"].get("stop_before_burnin_end") or 0)
    shift_std = np.full(p, float(shift["initial_std"]))
    scale = h.get("joint_scale_move") or {"enabled": False}
    scale_std = np.full(p, float(scale.get("initial_std", 0.1)))
    if resume is not None:             # continuation: no further adaptation
        adapt_end = -1
        shift_std = np.array(resume["shift_std"], dtype=float)
        scale_std = np.array(resume["scale_std"], dtype=float)
    sig_ub = float(h["sigma_upper_bound"])

    mu_h = np.empty((nsteps, p))
    sig_h = np.empty((nsteps, p))
    X_h = np.empty((nsteps, E, p))
    logpost = np.full(nsteps, np.nan)
    acc_shift = np.zeros((nsteps, p), dtype=bool)
    acc_scale = np.zeros((nsteps, p), dtype=bool)
    # Acceptance of each Metropolis block, recorded from the samplers themselves
    # (the chains cannot be used: the joint shift move also changes mu and X)
    acc_mu = np.zeros((nsteps, p), dtype=bool)
    acc_sig = np.zeros((nsteps, p), dtype=bool)
    acc_X = np.zeros((nsteps, E), dtype=bool)

    def total_loglik(Xc):
        return sum(lik(Xc[e]) for e, lik in enumerate(likelihoods))

    t0 = time.time()
    for i in tqdm(range(nsteps), desc=f"HC chain {chain + 1}", disable=not progress):
        for d in range(p):
            st = hyper[2 * d].step(st)
            acc_mu[i, d] = hyper[2 * d]._last_accepted
            st = hyper[2 * d + 1].step(st)
            acc_sig[i, d] = hyper[2 * d + 1]._last_accepted
        for e, smp in enumerate(x_samplers):
            st = smp.step(st)
            acc_X[i, e] = smp._last_accepted

        ll_cur = None
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
                # The hierarchical term is invariant under the shift: the acceptance
                # depends on the likelihood and, with a normal prior on mu, on its ratio.
                ll_new = total_loglik(X_new)
                dprior = normal_logprior(mu_new, prior_mu) - normal_logprior(st.mu, prior_mu)
                if np.log(rng.uniform()) < ll_new - ll_old + dprior:
                    st = states.state(mu_new, st.sig.copy(), X_new)
                    ll_old = ll_new
                    acc_shift[i, d] = True
            ll_cur = ll_old
            every = int(shift["adapt_every"])
            if i < adapt_end and (i + 1) % every == 0:
                rate = acc_shift[i + 1 - every:i + 1].mean(axis=0)
                shift_std *= np.exp(rate - float(shift["target_acceptance"]))

        if scale["enabled"]:
            # Group (scale) move of Liu & Sabatti (2000): lam = exp(eps), eps ~ N(0, s^2)
            # (symmetric in log lam); sigma_d -> lam sigma_d, X[:, d] -> mu_d + lam (X[:, d] - mu_d).
            # Hierarchical term ratio lam^(-E), Jacobian of the E + 1 scaled coordinates
            # lam^(E + 1), uniform prior on sigma: log acceptance = dloglik + log(lam).
            ll_old = ll_cur if ll_cur is not None else total_loglik(st.X)
            for d in range(p):
                eps = rng.normal(0.0, scale_std[d])
                lam = np.exp(eps)
                sig_new, X_new = st.sig.copy(), st.X.copy()
                sig_new[d] *= lam
                X_new[:, d] = st.mu[d] + lam * (st.X[:, d] - st.mu[d])
                inside = (0.0 < sig_new[d] <= sig_ub) and \
                    np.all((X_new[:, d] >= lower[d]) & (X_new[:, d] <= upper[d]))
                if not inside:
                    continue
                ll_new = total_loglik(X_new)
                if np.log(rng.uniform()) < ll_new - ll_old + eps:
                    st = states.state(st.mu.copy(), sig_new, X_new)
                    ll_old = ll_new
                    acc_scale[i, d] = True
            ll_cur = ll_old
            every = int(scale.get("adapt_every", 100))
            if i < adapt_end and (i + 1) % every == 0:
                rate = acc_scale[i + 1 - every:i + 1].mean(axis=0)
                scale_std *= np.exp(rate - float(scale.get("target_acceptance", 0.44)))

        mu_h[i], sig_h[i], X_h[i] = st.mu, st.sig, st.X
        if h["store_logpost"]:
            logpost[i] = logp_hyper(st) + normal_logprior(st.mu, prior_mu) + \
                (ll_cur if ll_cur is not None else total_loglik(st.X))

    checkpoint = {"mu": st.mu.copy(), "sig": st.sig.copy(), "X": st.X.copy(),
                  "rng_state": rng.bit_generator.state,
                  "proposals": [(np.array(smp.proposal.cov, dtype=float), float(smp.proposal.adaptation_factor))
                                for smp in hyper + x_samplers],
                  "shift_std": shift_std.copy(), "scale_std": scale_std.copy()}
    return {"mu": mu_h, "sigma": sig_h, "X": X_h, "logpost": logpost, "checkpoint": checkpoint,
            "acceptance_shift": acc_shift, "shift_std_final": shift_std,
            "acceptance_scale": acc_scale, "scale_std_final": scale_std,
            "acceptance_mu": acc_mu, "acceptance_sigma": acc_sig, "acceptance_X": acc_X,
            "wall_clock_s": time.time() - t0, "seed": seed, "burnin": burnin, "nsteps": nsteps}


def _run_chain_args(args):
    return run_chain(*args)


def run_hierarchical(likelihoods: list, case_ids: list, lower, upper, cfg: dict,
                     progress: bool = True, nsteps: int | None = None, previous: dict | None = None) -> dict:
    """Run ``n_chains`` independent chains (different seeds; dispersed starts if requested).

    With ``parallel_chains`` the chains run in separate processes, one per CPU core.
    ``nsteps`` overrides ``hierarchical.nsteps`` (used by the length escalation).
    ``previous``: result of a previous call. The chains are then CONTINUED from their last
    state up to ``nsteps`` iterations in total (``nsteps - previous["nsteps"]`` new
    iterations, proposals frozen), and the returned histories cover all ``nsteps``
    iterations; the burn-in is recomputed as ``burnin_fraction * nsteps``.
    """
    h = cfg["hierarchical"]
    n = int(h["n_chains"])
    total = int(nsteps or h["nsteps"])
    if previous is None and h.get("init_X", "dispersed") == "case_mode" and h.get("_case_modes") is None:
        modes, ll = case_modes(likelihoods, lower, upper, seed=int(cfg["project"]["seed"]))
        h["_case_modes"] = modes.tolist()
        at_bound = int(np.sum(np.any((modes <= lower + 1e-6) | (modes >= upper - 1e-6), axis=1)))
        print(f"init_X = case_mode: likelihood maximum of each of the {len(modes)} cases found "
              f"({at_bound} at a bound of the prior range); X starts there (jitter "
              f"{float(h.get('init_X_jitter', 0.05))}), mu and sigma dispersed between the chains")
    if previous is not None:
        n_new = total - int(previous["nsteps"])
        if n_new <= 0:
            return previous
        resumes = list(previous["checkpoints"])
    else:
        n_new, resumes = total, [None] * n
    jobs = [(c, likelihoods, case_ids, lower, upper, cfg, progress, n_new, resumes[c]) for c in range(n)]
    workers = min(n, os.cpu_count() or 1)
    if h.get("parallel_chains") and n > 1 and workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        print(f"Running {n} chains on {workers} CPU cores ...")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            chains = list(pool.map(_run_chain_args, [j[:6] + (False,) + j[7:] for j in jobs]))
    else:
        chains = [run_chain(*j) for j in jobs]
    stack = lambda k: np.stack([ch[k] for ch in chains])   # noqa: E731
    out = {k: stack(k) for k in ("mu", "sigma", "X", "logpost", "acceptance_shift", "shift_std_final",
                                 "acceptance_scale", "scale_std_final",
                                 "acceptance_mu", "acceptance_sigma", "acceptance_X")}
    out.update({"wall_clock_s": np.array([ch["wall_clock_s"] for ch in chains]),
                "seeds": np.array([ch["seed"] for ch in chains]),
                "checkpoints": [ch["checkpoint"] for ch in chains],
                "burnin": chains[0]["burnin"], "nsteps": chains[0]["nsteps"], "continued_from": 0})
    if previous is None:
        return out
    # Concatenate the new iterations to the previous ones (iteration axis = 1)
    for k in _CONCAT_KEYS:
        out[k] = np.concatenate([previous[k], out[k]], axis=1)
    out["wall_clock_s"] = previous["wall_clock_s"] + out["wall_clock_s"]
    out["nsteps"] = total
    out["burnin"] = int(round(h["burnin_fraction"] * total))
    out["continued_from"] = int(previous["nsteps"])
    return out


# --- checkpoints in the results file (continuation in a later run) -----------------------

_CONCAT_KEYS = ("mu", "sigma", "X", "logpost", "acceptance_shift", "acceptance_scale",
                "acceptance_mu", "acceptance_sigma", "acceptance_X")


def checkpoints_for_h5(checkpoints: list) -> dict:
    """Checkpoints of the chains as a nested dict of arrays (``io.save_results``)."""
    import json
    out = {}
    for c, ck in enumerate(checkpoints):
        g = {"mu": np.asarray(ck["mu"], float), "sig": np.asarray(ck["sig"], float),
             "X": np.asarray(ck["X"], float), "shift_std": np.asarray(ck["shift_std"], float),
             "scale_std": np.asarray(ck["scale_std"], float),
             "rng_state": json.dumps(ck["rng_state"]),
             "prop_factor": np.array([f for _, f in ck["proposals"]], dtype=float)}
        for k, (cov, _) in enumerate(ck["proposals"]):
            g[f"prop_cov_{k:04d}"] = np.atleast_2d(np.asarray(cov, float))
        out[f"chain{c:02d}"] = g
    return out


def previous_from_saved(saved: dict) -> dict:
    """Result dict of ``run_hierarchical`` rebuilt from a saved HC_*.h5 (``io.load_results``),
    to be passed as ``previous``. Raises ValueError if the file has no checkpoints (files
    written before the continuation feature): such chains cannot be continued exactly."""
    import json
    if "checkpoints" not in saved:
        raise ValueError("this HC file has no chain checkpoints (written by an older version): "
                         "it cannot be continued, run the hierarchical calibration again.")
    ch = saved["chains"]
    prev = {k: np.asarray(ch[k], dtype=bool if k.startswith("acceptance") else float)
            for k in _CONCAT_KEYS}
    cks = []
    for name in sorted(saved["checkpoints"]):
        g = saved["checkpoints"][name]
        covs = [g[k] for k in sorted(g) if k.startswith("prop_cov_")]
        cks.append({"mu": g["mu"], "sig": g["sig"], "X": g["X"], "shift_std": g["shift_std"],
                    "scale_std": g["scale_std"], "rng_state": json.loads(g["rng_state"]),
                    "proposals": list(zip(covs, g["prop_factor"]))})
    md = saved["metadata"]
    prev.update({"checkpoints": cks, "nsteps": int(md["nsteps"]), "burnin": int(md["burnin"]),
                 "wall_clock_s": np.asarray(md["wall_clock_s"], float),
                 "seeds": np.asarray(md["seeds"]), "continued_from": 0,
                 "shift_std_final": ch["shift_std_final"], "scale_std_final": ch["scale_std_final"]})
    return prev

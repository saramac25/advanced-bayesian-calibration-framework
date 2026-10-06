"""Step 06 - convergence diagnostics (works for SC and HC, scalar or
time-dependent QoI: the method is read from the results file).

Criteria (config, block `diagnostics`): R-hat < rhat_max, bulk ESS >=
ess_per_chain * number of chains (walkers for emcee), acceptance rate in
range, MCSE/sigma (overlapping batch means; checked only if
mcse_over_std_max is set), wall-clock time.

Hierarchical calibration: one table with every block of the sampler,
mu_d, sigma_d and X[case, d], plus the acceptance of the joint shift
move, an adaptation-collapse check (with a suggested fixed proposal
covariance) and a NaN check. Every failed criterion comes with a
SUGGESTION.

  python workflows/06_diagnostics.py config.yaml            (latest SC and HC runs)
  python workflows/06_diagnostics.py config.yaml --file results/calibration/HC_....h5
"""

from pathlib import Path

import numpy as np
import pandas as pd

from abcf.config import load_config, output_dir
from abcf.diagnostics import sampler_health as health
from abcf.diagnostics.advice import hierarchical_advice, standard_advice
from abcf.diagnostics.convergence import acceptance_from_chain, chain_diagnostics
from abcf.io import load_results
from abcf.workflow import latest_result, parse_args

pd.set_option("display.width", 200)


def criteria_ok(row, cfg, n_chains):
    d = cfg["diagnostics"]
    ok = row["rhat"] < d["rhat_max"] and row["ess_bulk"] >= d["ess_per_chain"] * n_chains
    lo, hi = row["acceptance_range"]
    if row.get("acceptance") is not None and not np.isnan(row["acceptance"]):
        ok = ok and lo <= row["acceptance"] <= hi
    if d["mcse_over_std_max"] is not None:
        ok = ok and row["mcse_over_std"] <= d["mcse_over_std_max"]
    return bool(ok)


def block_rows(chains, names, block, case, acceptance, acc_range, cfg, n_chains):
    """Rows for a block; chains (n_chains, n_draws, p), acceptance scalar or (p,)."""
    dg = chain_diagnostics(chains)
    rows = []
    for j, n in enumerate(names):
        acc = acceptance[j] if np.ndim(acceptance) else acceptance
        row = {"block": block, "case": case, "parameter": n,
               **{k: float(v[j]) for k, v in dg.items()},
               "ess_threshold": cfg["diagnostics"]["ess_per_chain"] * n_chains,
               "acceptance": float(acc), "acceptance_range": tuple(acc_range)}
        row["pass"] = criteria_ok(row, cfg, n_chains)
        rows.append(row)
    return rows


def diagnose_standard(r, cfg, out, stem):
    md, ch = r["metadata"], r["chains"]
    names, burnin = list(md["param_names"]), int(md["burnin"])
    theta = ch["theta"][:, burnin:, :]
    n_walkers = theta.shape[0]
    acc = float(np.mean(ch["acceptance"]))
    rows = block_rows(theta, names, "theta", "", acc, cfg["diagnostics"]["acceptance_range_standard"],
                      cfg, n_walkers)
    df = pd.DataFrame(rows)
    print(f"walkers = {n_walkers}, steps = {md['nsteps']}, burn-in = {burnin}, "
          f"wall clock = {float(np.sum(md['wall_clock_s'])):.1f} s")
    advice = standard_advice(acc, df["rhat"].max(), df["ess_bulk"].min(), n_walkers, int(md["nsteps"]), cfg)
    return df, advice


def diagnose_hierarchical(r, cfg, out, stem):
    md, ch = r["metadata"], r["chains"]
    names, burnin, nsteps = list(md["param_names"]), int(md["burnin"]), int(md["nsteps"])
    cases = list(md["case_ids"])
    d = cfg["diagnostics"]
    mu, sig, X = ch["mu"], ch["sigma"], ch["X"]
    n_chains, p = mu.shape[0], mu.shape[2]
    range_1d = d["acceptance_range_1d"]
    range_X = range_1d if p == 1 else d["acceptance_range_X"]

    recorded = "acceptance_mu" in ch
    if recorded:
        acc_mu = ch["acceptance_mu"][:, burnin:, :].mean(axis=(0, 1))
        acc_sig = ch["acceptance_sigma"][:, burnin:, :].mean(axis=(0, 1))
        acc_X = ch["acceptance_X"][:, burnin:, :].mean(axis=(0, 1))
    else:   # results written by an older version: estimate from the chains
        acc_mu = np.array([np.mean([acceptance_from_chain(mu[c, burnin:, j]) for c in range(n_chains)])
                           for j in range(p)])
        acc_sig = np.array([np.mean([acceptance_from_chain(sig[c, burnin:, j]) for c in range(n_chains)])
                            for j in range(p)])
        acc_X = np.array([np.mean([acceptance_from_chain(X[c, burnin:, e, :]) for c in range(n_chains)])
                          for e in range(len(cases))])
        print("NOTE: acceptance estimated from the chains (older results file); it is inflated "
              "if the joint shift move was used.")

    rows = block_rows(mu[:, burnin:], names, "mu", "", acc_mu, range_1d, cfg, n_chains)
    rows += block_rows(sig[:, burnin:], names, "sigma", "", acc_sig, range_1d, cfg, n_chains)
    for e, cid in enumerate(cases):
        rows += block_rows(X[:, burnin:, e, :], names, "X", cid, acc_X[e], range_X, cfg, n_chains)
    df = pd.DataFrame(rows)

    print(f"chains = {n_chains}, steps = {nsteps}, burn-in = {burnin}, "
          f"wall clock per chain = {np.round(np.atleast_1d(md['wall_clock_s']), 1)} s")
    if md.get("joint_shift_move") and "acceptance_shift" in ch:
        acc_shift = ch["acceptance_shift"][:, burnin:, :].mean(axis=(0, 1))
        print("Joint shift move acceptance: " + ", ".join(f"{n} {a:.2f}" for n, a in zip(names, acc_shift)))

    coll, sugg = health.collapse_check(X, nsteps, cases, collapse_ar=d["collapse_ratio"],
                                       acc=ch["acceptance_X"] if recorded else None)
    coll.to_csv(out / f"{stem}_collapse_check.csv", index=False)
    if coll["collapsed"].any():
        print("Adaptation collapse detected for: "
              f"{sorted(set(coll.loc[coll['collapsed'], 'case']))}. SUGGESTION, add to the config:")
        print(health.format_fixed_cov_yaml(sugg))
    nans = health.nan_check(X, cases)
    print("NaN in the chains: " + (str(nans) if nans else "none"))

    advice = []
    for row in rows:
        if not row["pass"]:
            label = f"{row['block']}[{row['case'] + ', ' if row['case'] else ''}{row['parameter']}]"
            advice += [f"{label}: {a}" for a in hierarchical_advice(row, cfg, n_chains, nsteps)]
    return df, advice


def diagnose(path: Path, cfg):
    r = load_results(path)
    md = r["metadata"]
    out = output_dir(cfg, "diagnostics")
    stem = path.stem
    print(f"\n=== {stem} ({md['method']}, {md['qoi_type']} QoI) ===")
    if md["method"] == "SC":
        df, advice = diagnose_standard(r, cfg, out, stem)
    else:
        df, advice = diagnose_hierarchical(r, cfg, out, stem)
    df["wall_clock_s_total"] = float(np.sum(md["wall_clock_s"]))
    df.to_csv(out / f"{stem}_convergence.csv", index=False)
    cols = ["block", "case", "parameter", "rhat", "ess_bulk", "ess_tail", "ess_threshold",
            "mcse_over_std", "acceptance", "pass"]
    show = df[cols] if df["case"].astype(bool).any() else df[[c for c in cols if c != "case"]]
    print(show.to_string(index=False, float_format="%.4g"))
    if df["pass"].all():
        print("All criteria met.")
    else:
        print(f"{int((~df['pass']).sum())}/{len(df)} rows do not meet all criteria.")
        for a in dict.fromkeys(advice):
            print(f"  SUGGESTION: {a}")


def main():
    args = parse_args(__doc__, lambda p: p.add_argument("--file", default=None))
    cfg = load_config(args.config)
    files = [Path(args.file)] if args.file else [f for f in (latest_result(cfg, "SC"),
                                                             latest_result(cfg, "HC")) if f]
    if not files:
        raise FileNotFoundError("No calibration results found: run step 04 and/or 05 first.")
    for f in files:
        diagnose(f, cfg)


if __name__ == "__main__":
    main()

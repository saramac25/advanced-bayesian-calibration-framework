"""Step 06 - convergence diagnostics (works for SC and HC, scalar or
time-dependent QoI: the method is read from the results file).

Criteria (config, block `diagnostics`, see abcf/diagnostics/criteria.py):
R-hat <= rhat_max, bulk ESS >= ess_per_chain * number of chains (walkers
for emcee; for emcee optionally N * n_walkers / tau), tail-ESS (if
check_tail_ess), N >= tau_factor * tau (if tau_factor), acceptance rate in
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
from abcf.diagnostics.convergence import acceptance_from_chain
from abcf.diagnostics.criteria import criteria_rows, hierarchical_rows, standard_rows
from abcf.io import load_results
from abcf.workflow import latest_result, parse_args

pd.set_option("display.width", 200)


def diagnose_standard(r, cfg, out, stem):
    md, ch = r["metadata"], r["chains"]
    names, burnin = list(md["param_names"]), int(md["burnin"])
    theta = ch["theta"][:, burnin:, :]
    n_walkers = theta.shape[0]
    acc = float(np.mean(ch["acceptance"]))
    rows = standard_rows(r, cfg, names)
    df = pd.DataFrame(rows)
    print(f"walkers = {n_walkers}, steps = {md['nsteps']}, burn-in = {burnin}, "
          f"wall clock = {float(np.sum(md['wall_clock_s'])):.1f} s")
    rhat = df["rhat"].max() if cfg["diagnostics"].get("standard_rhat", True) else 0.0
    advice = standard_advice(acc, rhat, df["ess_bulk"].min(), n_walkers, int(md["nsteps"]), cfg)
    if not df["tau_ok"].all():
        advice.append(f"N = {theta.shape[1]} post-burn-in steps < {cfg['diagnostics']['tau_factor']} tau "
                      f"(tau = {df['tau'].max():.1f}): run longer.")
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
        rows = hierarchical_rows(ch, burnin, names, cases, cfg)
    else:   # results written by an older version: estimate from the chains
        acc_mu = np.array([np.mean([acceptance_from_chain(mu[c, burnin:, j]) for c in range(n_chains)])
                           for j in range(p)])
        acc_sig = np.array([np.mean([acceptance_from_chain(sig[c, burnin:, j]) for c in range(n_chains)])
                            for j in range(p)])
        acc_X = np.array([np.mean([acceptance_from_chain(X[c, burnin:, e, :]) for c in range(n_chains)])
                          for e in range(len(cases))])
        print("NOTE: acceptance estimated from the chains (older results file); it is inflated "
              "if the joint shift move was used.")
        rows = criteria_rows(mu[:, burnin:], names, cfg, "HC", "mu", "", acc_mu, range_1d)
        rows += criteria_rows(sig[:, burnin:], names, cfg, "HC", "sigma", "", acc_sig, range_1d)
        for e, cid in enumerate(cases):
            rows += criteria_rows(X[:, burnin:, e, :], names, cfg, "HC", "X", cid, acc_X[e], range_X)
    df = pd.DataFrame(rows)

    print(f"chains = {n_chains}, steps = {nsteps}, burn-in = {burnin}, "
          f"wall clock per chain = {np.round(np.atleast_1d(md['wall_clock_s']), 1)} s")
    if md.get("joint_shift_move") and "acceptance_shift" in ch:
        acc_shift = ch["acceptance_shift"][:, burnin:, :].mean(axis=(0, 1))
        print("Joint shift move acceptance: " + ", ".join(f"{n} {a:.2f}" for n, a in zip(names, acc_shift)))
    if md.get("joint_scale_move") and "acceptance_scale" in ch:
        acc_scale = ch["acceptance_scale"][:, burnin:, :].mean(axis=(0, 1))
        print("Joint scale move acceptance: " + ", ".join(f"{n} {a:.2f}" for n, a in zip(names, acc_scale)))

    coll, sugg = health.collapse_check(X, nsteps, cases, collapse_ar=d["collapse_ratio"],
                                       acc=ch["acceptance_X"] if recorded else None)
    coll.to_csv(out / f"{stem}_collapse_check.csv", index=False)
    if coll["collapsed"].any():
        print("Adaptation collapse detected for: "
              f"{sorted(set(coll.loc[coll['collapsed'], 'case']))}. SUGGESTION, add to the config:")
        print(health.format_fixed_cov_yaml(sugg))
    nans = health.nan_check(X, cases)
    print("NaN in the chains: " + (str(nans) if nans else "none"))

    if "X" in set(df["block"]) and not df.loc[df["block"] == "X", "convergence_required"].all():
        x = df[df["block"] == "X"]
        print(f"X blocks (reported only): worst R-hat {x['rhat'].max():.4f}, "
              f"min bulk-ESS {x['ess_bulk'].min():.0f}, min tail-ESS {x['ess_tail'].min():.0f}, "
              f"acceptance {x['acceptance'].min():.2f}-{x['acceptance'].max():.2f}")
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
            "tau", "n_over_tau", "mcse_over_std", "acceptance", "converged", "pass"]
    show = df[cols] if df["case"].astype(bool).any() else df[[c for c in cols if c != "case"]]
    if len(show) > 40:      # many cases: print the mu/sigma rows and the failing rows only
        show = show[(df["block"] != "X") | (~df["pass"])]
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

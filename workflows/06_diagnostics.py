"""Step 06 - convergence diagnostics (works for SC and HC, scalar or
time-dependent QoI: the method is read from the results file).

Criteria (config, block `diagnostics`): R-hat < rhat_max, bulk ESS >=
ess_per_chain * number of chains (walkers for emcee), MCSE/sigma
(overlapping batch means), acceptance rate, wall-clock time. For the
hierarchical sampler also: acceptance per case, adaptation collapse
(with a suggested fixed proposal covariance) and NaN check.

  python workflows/06_diagnostics.py config.yaml            (latest SC and HC runs)
  python workflows/06_diagnostics.py config.yaml --file results/calibration/HC_....h5
"""

from pathlib import Path

import numpy as np
import pandas as pd

from abcf.config import load_config, output_dir
from abcf.diagnostics.convergence import acceptance_from_chain, chain_diagnostics
from abcf.diagnostics import sampler_health as health
from abcf.io import load_results
from abcf.workflow import latest_result, parse_args


def verdict(row, cfg, n_chains):
    d = cfg["diagnostics"]
    ok = row["rhat"] < d["rhat_max"] and row["ess_bulk"] >= d["ess_per_chain"] * n_chains
    if d["mcse_over_std_max"] is not None:
        ok = ok and row["mcse_over_std"] <= d["mcse_over_std_max"]
    return bool(ok)


def table(chains, names, block, cfg, n_chains):
    dg = chain_diagnostics(chains)
    rows = []
    for j, n in enumerate(names):
        row = {"block": block, "parameter": n, **{k: float(v[j]) for k, v in dg.items()},
               "ess_threshold": cfg["diagnostics"]["ess_per_chain"] * n_chains}
        row["pass"] = verdict(row, cfg, n_chains)
        rows.append(row)
    return rows


def diagnose(path: Path, cfg):
    r = load_results(path)
    md, ch = r["metadata"], r["chains"]
    names = list(md["param_names"])
    burnin = int(md["burnin"])
    out = output_dir(cfg, "diagnostics")
    stem = path.stem
    wall = np.atleast_1d(md["wall_clock_s"])
    print(f"\n=== {stem} ({md['method']}, {md['qoi_type']} QoI) ===")

    if md["method"] == "SC":
        theta = ch["theta"][:, burnin:, :]
        n_chains = theta.shape[0]
        rows = table(theta, names, "theta", cfg, n_chains)
        acc = float(np.mean(ch["acceptance"]))
        lo, hi = cfg["diagnostics"]["acceptance_range_standard"]
        print(f"walkers = {n_chains}, steps = {md['nsteps']}, burn-in = {burnin}, "
              f"acceptance = {acc:.3f} ({'ok' if lo <= acc <= hi else 'OUT of range'} [{lo}, {hi}]), "
              f"wall clock = {wall.sum():.1f} s")
    else:
        mu, sig, X = ch["mu"][:, burnin:, :], ch["sigma"][:, burnin:, :], ch["X"]
        n_chains = mu.shape[0]
        rows = table(mu, names, "mu", cfg, n_chains) + table(sig, names, "sigma", cfg, n_chains)
        for b, arr in (("mu", ch["mu"]), ("sigma", ch["sigma"])):
            for j, n in enumerate(names):
                ar = np.mean([acceptance_from_chain(arr[c, burnin:, j]) for c in range(n_chains)])
                for row in rows:
                    if row["block"] == b and row["parameter"] == n:
                        row["acceptance"] = ar
        cases = list(md["case_ids"])
        acc = health.acceptance_table(X, burnin, cases, names,
                                      cfg["diagnostics"]["acceptance_range_hierarchical"])
        acc.to_csv(out / f"{stem}_acceptance_X.csv", index=False)
        coll, sugg = health.collapse_check(X, int(md["nsteps"]), cases,
                                           collapse_ar=cfg["diagnostics"]["collapse_ratio"])
        coll.to_csv(out / f"{stem}_collapse_check.csv", index=False)
        nans = health.nan_check(X, cases)
        x_rows = []
        for e, cid in enumerate(cases):
            x_rows += [dict(r, case=cid) for r in table(X[:, burnin:, e, :], names, "X", cfg, n_chains)]
        pd.DataFrame(x_rows).to_csv(out / f"{stem}_convergence_X.csv", index=False)
        print(f"chains = {n_chains}, steps = {md['nsteps']}, burn-in = {burnin}, "
              f"wall clock per chain = {np.round(wall, 1)} s")
        bad = acc[~acc["ok"]]
        print(f"X blocks with acceptance out of range: {len(bad)}/{len(acc)}")
        if coll["collapsed"].any():
            print("Adaptation collapse detected for: "
                  f"{sorted(set(coll.loc[coll['collapsed'], 'case']))}. Suggested config:")
            print(health.format_fixed_cov_yaml(sugg))
        print("NaN in the chains: " + (str(nans) if nans else "none"))

    df = pd.DataFrame(rows)
    df["wall_clock_s_total"] = float(wall.sum())
    df.to_csv(out / f"{stem}_convergence.csv", index=False)
    cols = [c for c in ["block", "parameter", "rhat", "ess_bulk", "ess_tail", "ess_threshold",
                        "mcse_over_std", "acceptance", "pass"] if c in df.columns]
    print(df[cols].to_string(index=False, float_format="%.4g"))
    if not df["pass"].all():
        print("WARNING: not all criteria are met; run longer chains or tune the proposals.")


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

"""Readable verdicts and suggestions for the user."""

from __future__ import annotations


def standard_advice(acceptance: float, rhat_max: float, ess_min: float, nwalkers: int,
                    nsteps: int, cfg: dict) -> list:
    """Suggestions for the emcee run (empty list if every criterion is met)."""
    d, s = cfg["diagnostics"], cfg["standard"]
    lo, hi = d["acceptance_range_standard"]
    a = float(s["stretch_a"])
    out = []
    if acceptance > hi:
        out.append(f"acceptance {acceptance:.3f} > {hi}: the moves are too small. "
                   f"Increase standard.stretch_a (now {a:g}), e.g. to {a + 1:g}-{a + 2:g}.")
    elif acceptance < lo:
        new = max(1.2, a - 1)
        out.append(f"acceptance {acceptance:.3f} < {lo}: the moves are too large. "
                   f"Decrease standard.stretch_a (now {a:g}), e.g. to {new:g}.")
    if rhat_max >= d["rhat_max"]:
        out.append(f"R-hat {rhat_max:.4f} >= {d['rhat_max']}: chains not mixed. "
                   "Use more steps (standard.nsteps_list) or more walkers (walker_multipliers).")
    need = d["ess_per_chain"] * nwalkers
    if ess_min < need:
        factor = need / max(ess_min, 1.0)
        out.append(f"ESS {ess_min:.0f} < {need}: about {factor:.1f}x more steps needed "
                   f"(~{int(nsteps * factor) + 1} steps per walker).")
    return out


def hierarchical_advice(row: dict, cfg: dict, n_chains: int, nsteps: int) -> list:
    """Suggestions for one block (mu_d, sigma_d or X[case, d]) of the hierarchical sampler."""
    d = cfg["diagnostics"]
    out = []
    lo, hi = row["acceptance_range"]
    acc = row.get("acceptance")
    block = row["block"]
    key = "X" if block.startswith("X") else ("mu" if block == "mu" else "sigma")
    kind = cfg["hierarchical"]["adaptation"][key]
    if acc is not None and acc == acc:   # not NaN
        if acc > hi:
            out.append(f"acceptance {acc:.2f} > {hi}: proposal too narrow"
                       + (" (covariance adaptation: try s_d larger or acceptance_rate)."
                          if kind == "covariance" else
                          " (the scale adaptation did not reach its target: increase "
                          "hierarchical.adaptation.rate or the burn-in)."))
        elif acc < lo:
            out.append(f"acceptance {acc:.2f} < {lo}: proposal too wide"
                       + (" (covariance adaptation: try s_d smaller or acceptance_rate)."
                          if kind == "covariance" else
                          " (the scale adaptation did not reach its target: increase "
                          "hierarchical.adaptation.rate or the burn-in)."))
    if row["rhat"] >= d["rhat_max"]:
        out.append(f"R-hat {row['rhat']:.3f}: run longer chains (hierarchical.nsteps).")
    need = d["ess_per_chain"] * n_chains
    if row["ess_bulk"] < need:
        out.append(f"ESS {row['ess_bulk']:.0f} < {need}: ~{need / max(row['ess_bulk'], 1):.1f}x more steps "
                   f"(~{int(nsteps * need / max(row['ess_bulk'], 1)) + 1}).")
    return out

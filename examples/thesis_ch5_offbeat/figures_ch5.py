"""Figures and summary tables of the Chapter 5 scalar case studies.

For each case (epistemic, aleatoric, model bias), from the latest standard
(SC) and hierarchical (HC) results of the case:

  parameters_<case>.png/pdf   prior, standard posterior and hierarchical posterior
                              (population of a new rod) in the same panel; true
                              values as rug lines; UO2 k left, UO2 rho right
  traces_<case>.png/pdf       emcee walkers (theta) and hierarchical chains (mu, sigma)
  predictive_<case>.png/pdf   95% posterior predictive intervals: calibration rods
                              (relative deviation from the measurement) and the 6
                              validation rods (FCT), with chi2_E and chi2_tot

and the tables summary_posterior.csv, summary_validation.csv,
summary_convergence.csv in results/figures/ (all cases together).

Called by run_thesis_ch5.py (stage "figures"); can also be run alone:
    python figures_ch5.py [results folder]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.stats import gaussian_kde, norm  # noqa: E402

from abcf.config import case_noise, validation_case_list  # noqa: E402
from abcf.io import load_results  # noqa: E402
from abcf.postprocessing.posterior import Posterior  # noqa: E402
from abcf.workflow import calibration_cases, latest_result, load_all  # noqa: E402

HERE = Path(__file__).resolve().parent
CASES = ["epistemic", "aleatoric", "modelbias_uniform_k", "modelbias_struct_k", "epistemic_prior"]
CASE_TITLE = {"epistemic": "Epistemic uncertainty", "aleatoric": "Aleatoric uncertainty",
              "modelbias_uniform_k": "Model bias common to all rods",
              "modelbias_struct_k": "Model bias dependent on the rod power",
              "epistemic_prior": "Epistemic uncertainty, literature priors"}
RUG_H, RUG_LW = 0.25, 1.5     # true-value rug lines: height (fraction of the y axis) and width, all cases
ORDER = ["UO2_k", "UO2_alphaT"]                   # k left, alphaT right
LABEL = {}   # display names, from the labels block of the configurations (load_case)

C_SC, C_HC, C_TRUE, C_PRIOR, C_EXP, C_FILL = "#7b3294", "#e69f00", "#0072b2", "0.55", "#0b2c5a", "#cfe4f1"
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 13,
                     "axes.titlesize": 14, "axes.labelsize": 14, "legend.fontsize": 11,
                     "axes.linewidth": 1.1})
N_DRAWS = 4000


# --------------------------------------------------------------------------- #
#  Loading
# --------------------------------------------------------------------------- #
def load_case(config: Path) -> dict:
    cfg, data, setup, sset = load_all(config)
    files = {m: latest_result(cfg, m) for m in ("SC", "HC")}
    missing = [m for m, f in files.items() if f is None]
    if missing:
        raise FileNotFoundError(f"{config.name}: no {missing} results (run the calibration first).")
    res = {m: load_results(f) for m, f in files.items()}
    post = {m: Posterior(r) for m, r in res.items()}
    meta = json.loads((HERE / "inputs" / "metadata.json").read_text())
    LABEL.update(cfg.get("labels") or {})
    exp = pd.read_csv(HERE / "inputs" / "synthetic_experiments.csv")
    return {"cfg": cfg, "data": data, "setup": setup, "sset": sset, "res": res, "post": post,
            "files": files, "meta": meta, "exp": exp,
            "cal": sorted(calibration_cases(cfg, data), key=_rod_number),
            "val": sorted(validation_case_list(cfg), key=_rod_number)}


def _rod_number(cid: str):
    digits = "".join(ch for ch in cid if ch.isdigit())
    return int(digits) if digits else cid


def dataset_of(name: str) -> str:
    return "B" if name == "aleatoric" else "A"


def truths(c: dict, name: str, param: str):
    """(single true value or None, true values of the calibration rods, population (mu, sd) or None)."""
    exp = c["exp"][c["exp"]["dataset"] == dataset_of(name)].set_index("rod")
    rods = exp.loc[c["cal"], param].to_numpy(dtype=float)
    if name == "aleatoric":
        d = c["meta"]["distributions"][param]
        return None, rods, (d["mean"], d["std"])
    return float(c["meta"]["true_A"][param]), rods, None


def bounded_kde(samples, lo, hi, n=800):
    """Gaussian KDE reflected at the prior bounds (no mass outside the prior)."""
    kde = gaussian_kde(samples)
    xx = np.linspace(lo, hi, n)
    return xx, kde(xx) + kde(2 * lo - xx) + kde(2 * hi - xx)


def samples_of(c: dict, method: str, n: int = 20000) -> np.ndarray:
    seed = int(c["cfg"]["project"]["seed"])
    p = c["post"][method]
    return p.theta if method == "SC" else p.population(n, seed)


# --------------------------------------------------------------------------- #
#  Figure 1: prior, standard and hierarchical posteriors, true values
# --------------------------------------------------------------------------- #
def fig_parameters(c: dict, name: str, out: Path) -> list:
    post_sc = c["post"]["SC"]
    names = post_sc.names
    order = [q for q in ORDER if q in names]          # model-bias cases: k only
    fig, axes = plt.subplots(1, len(order), figsize=(6.5 * len(order), 4.8), squeeze=False)
    axes = axes[0]
    rows = []
    for ax, param in zip(axes, order):
        j = names.index(param)
        lo, hi = post_sc.lower_phys[j], post_sc.upper_phys[j]
        pr = (c["cfg"]["parameters"][param].get("prior") or {})
        if pr.get("type") == "normal":            # normal prior truncated to the GP range
            from scipy.stats import truncnorm
            m_, s_ = float(pr["mean"]), float(pr["std"])
            xg = np.linspace(lo, hi, 1000)
            pg = truncnorm.pdf(xg, (lo - m_) / s_, (hi - m_) / s_, loc=m_, scale=s_)
            ax.plot(xg, pg, color=C_PRIOR, lw=2, ls="--", zorder=3)
            p_prior = pg.max()
        else:
            p_prior = 1.0 / (hi - lo)
            ax.hlines(p_prior, lo, hi, color=C_PRIOR, lw=2, ls="--", zorder=3)
        peaks = []
        for m, col in (("SC", C_SC), ("HC", C_HC)):
            s = samples_of(c, m)[:, j]
            xx, yy = bounded_kde(s, lo, hi)
            peaks.append(yy.max())
            ax.fill_between(xx, yy, color=col, alpha=0.3, lw=0, zorder=3)
            ax.plot(xx, yy, color=col, lw=2.2, zorder=4)
            q = np.percentile(s, [2.5, 97.5])
            single, rods, pop = truths(c, name, param)
            ref = single if single is not None else pop[0]
            rows.append({"case": name, "method": m, "parameter": param, "mean": s.mean(),
                         "std": s.std(), "q2.5": q[0], "q97.5": q[1],
                         "true_value_or_population_mean": ref,
                         "true_in_CI95": bool(q[0] <= ref <= q[1]),
                         "rods_in_CI95_%": 100 * np.mean((rods >= q[0]) & (rods <= q[1]))})
            if m == "HC":
                p = c["post"]["HC"]
                rows[-1].update({"mu_mean": p.mu[:, j].mean(), "mu_q2.5": np.percentile(p.mu[:, j], 2.5),
                                 "mu_q97.5": np.percentile(p.mu[:, j], 97.5),
                                 "sigma_mean": p.sigma[:, j].mean(),
                                 "sigma_q2.5": np.percentile(p.sigma[:, j], 2.5),
                                 "sigma_q97.5": np.percentile(p.sigma[:, j], 97.5)})
        ymax = 1.15 * max(max(peaks), p_prior)
        single, rods, pop = truths(c, name, param)
        if pop is not None:                       # true population (rescaled if it squashes the plot)
            xg = np.linspace(lo, hi, 1500)
            pg = norm.pdf(xg, *pop)
            if pg.max() > 1.5 * ymax:
                pg = pg / pg.max() * 0.9 * ymax
                ax.text(0.98, 0.97, "true pdf rescaled", transform=ax.transAxes, ha="right", va="top",
                        fontsize=9, color=C_TRUE, style="italic")
            ax.fill_between(xg, pg, color=C_FILL, alpha=0.5, lw=0, zorder=1.5)
            ax.plot(xg, pg, color=C_TRUE, lw=1.8, zorder=5)
            ax.vlines(rods, 0, RUG_H, color=C_TRUE, lw=RUG_LW, zorder=6, transform=ax.get_xaxis_transform())
        else:                                     # single true value: same rug line
            ax.vlines(single, 0, RUG_H, color=C_TRUE, lw=RUG_LW, zorder=6, transform=ax.get_xaxis_transform())
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, ymax)
        ax.set_xlabel(LABEL.get(param, param) + " (multiplier)")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("PDF")
    handles = [Line2D([0], [0], color=C_PRIOR, lw=2, ls="--"),
               (Patch(color=C_SC, alpha=0.3), Line2D([0], [0], color=C_SC, lw=2.2)),
               (Patch(color=C_HC, alpha=0.3), Line2D([0], [0], color=C_HC, lw=2.2))]
    normal_pr = any((c["cfg"]["parameters"][q].get("prior") or {}).get("type") == "normal" for q in order)
    labels = ["prior (normal, truncated to the GP range)" if normal_pr else "uniform prior (GP training range)",
              "standard calibration: posterior",
              "hierarchical calibration: population (new rod)"]
    if name == "aleatoric":
        handles += [(Patch(color=C_FILL), Line2D([0], [0], color=C_TRUE, lw=1.8)),
                    Line2D([0], [0], marker="|", ms=14, mew=RUG_LW, lw=0, color=C_TRUE)]
        labels += [r"true population $\mathcal{N}(\mu,\sigma^2)$",
                   f"true values of the {len(c['cal'])} calibration rods"]
    else:
        handles.append(Line2D([0], [0], marker="|", ms=14, mew=RUG_LW, lw=0, color=C_TRUE))
        labels.append("true value (same for all rods)")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=1 if len(order) == 1 else 2,
               frameon=False)
    fig.suptitle(CASE_TITLE[name])
    fig.tight_layout()
    _save(fig, out / f"parameters_{name}")
    return rows


# --------------------------------------------------------------------------- #
#  Figure 2: trace plots
# --------------------------------------------------------------------------- #
def fig_traces(c: dict, name: str, out: Path):
    sc, hc = c["res"]["SC"], c["res"]["HC"]
    names = list(sc["metadata"]["param_names"])
    idx = np.asarray(sc["metadata"]["calibrated_idx"], dtype=int)
    scal = sc["scaler"]
    order = [q for q in ORDER if q in names]
    fig, axes = plt.subplots(len(order), 3, figsize=(16, 3.25 * len(order)), sharex="col", squeeze=False)
    for r, param in enumerate(order):
        j = names.index(param)
        mean, std = scal["mean"][idx[j]], scal["std"][idx[j]]
        single, rods, pop = truths(c, name, param)
        panels = [(sc["chains"]["theta"], int(sc["metadata"]["burnin"]), r"standard $\theta$", mean, std),
                  (hc["chains"]["mu"], int(hc["metadata"]["burnin"]), r"hierarchical $\mu$", mean, std),
                  (hc["chains"]["sigma"], int(hc["metadata"]["burnin"]), r"hierarchical $\sigma$", 0.0, std)]
        for k, (arr, burnin, title, shift, scale) in enumerate(panels):
            ax = axes[r, k]
            for ch in range(arr.shape[0]):
                ax.plot(arr[ch, :, j] * scale + shift, lw=0.4, alpha=0.8)
            ax.axvline(burnin, color="gray", ls="--", lw=1)
            true = (pop[1] if pop else 0.0) if k == 2 else (pop[0] if pop else single)
            ax.axhline(true, color="k", lw=1.3)
            if r == 0:
                ax.set_title(title)
            if k == 0:
                ax.set_ylabel(LABEL.get(param, param))
    for ax in axes[-1]:
        ax.set_xlabel("iteration")
    fig.legend([Line2D([0], [0], color="gray", ls="--"), Line2D([0], [0], color="k", lw=1.3)],
               ["end of burn-in", "true value (population mean / std for the aleatoric case)"],
               loc="upper center", bbox_to_anchor=(0.5, 0.01), ncol=2, frameon=False)
    fig.suptitle(f"{CASE_TITLE[name]}: chains (4 walkers / 4 chains)")
    fig.tight_layout()
    _save(fig, out / f"traces_{name}", dpi=130)


# --------------------------------------------------------------------------- #
#  Figure 3: posterior predictive, calibration and validation rods
# --------------------------------------------------------------------------- #
def _predictive(c: dict, method: str, cid: str, case_index: int | None, n: int, seed: int):
    """Predictive samples of the FCT of one rod, GP variance included."""
    p = c["post"][method]
    if case_index is None:        # validation rod: SC posterior / HC new-rod population
        theta = p.draws(n, seed)
    else:                         # calibration rod: SC posterior / HC X[rod]
        theta = p.case_draws(case_index, n, seed)
    sur, sset = c["sset"][cid], c["sset"]
    m, v = sur.predict(sset.scaler.transform(c["setup"].full_vector(theta, cid)))
    rng = np.random.default_rng(seed)
    return m[:, 0] + rng.standard_normal(len(m)) * np.sqrt(np.maximum(v[:, 0], 0.0))


def fig_predictive(c: dict, name: str, out: Path, vsum: pd.DataFrame | None):
    seed = int(c["cfg"]["project"]["seed"])
    data, cal, val = c["data"], c["cal"], c["val"]
    hc_cases = list(c["post"]["HC"].case_ids)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(17, 5.2), gridspec_kw={"width_ratios": [3.2, 1]})
    # calibration rods: relative deviation from the measurement [%]
    x = np.arange(len(cal))
    rel_noise = case_noise(c["cfg"], cal[0])
    a1.axhspan(-196 * rel_noise, 196 * rel_noise, color=C_EXP, alpha=0.12, lw=0,
               label=r"measurement $\pm1.96\,\sigma_{exp}$")
    a1.axhline(0, color=C_EXP, lw=0.8)
    for dx, m, col in ((-0.18, "SC", C_SC), (0.18, "HC", C_HC)):
        q = []
        for i, cid in enumerate(cal):
            y = data.cases[cid].meas[0]
            s = _predictive(c, m, cid, i if m == "SC" else hc_cases.index(cid), N_DRAWS, seed)
            q.append(100 * (np.percentile(s, [2.5, 50, 97.5]) - y) / y)
        q = np.array(q)
        a1.vlines(x + dx, q[:, 0], q[:, 2], color=col, lw=1.4,
                  label=f"95% predictive interval: {'standard' if m == 'SC' else 'hierarchical'}")
        a1.plot(x + dx, q[:, 1], "o", ms=2.5, color=col)
    a1.set_xlim(-1, len(cal))
    step = max(1, len(cal) // 25)
    a1.set_xticks(x[::step], [r.replace("ROD", "") for r in cal[::step]], fontsize=9)
    a1.set_xlabel("calibration rod")
    a1.set_ylabel(r"(prediction $-$ measurement) / measurement [%]")
    a1.set_title(f"{len(cal)} calibration rods")
    # validation rods: FCT
    xv = np.arange(len(val))
    yv = np.array([data.cases[r].meas[0] for r in val])
    a2.errorbar(xv - 0.22, yv, yerr=1.96 * rel_noise * yv, fmt="D", ms=6, color=C_EXP, capsize=4)
    for dx, m, col in ((0.0, "SC", C_SC), (0.22, "HC", C_HC)):
        q = np.array([np.percentile(_predictive(c, m, r, None, N_DRAWS, seed), [2.5, 50, 97.5]) for r in val])
        a2.errorbar(xv + dx, q[:, 1], yerr=[q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]], fmt="o", ms=5,
                    color=col, capsize=4)
    a2.set_xticks(xv, [r.replace("ROD", "") for r in val])
    a2.set_xlabel("validation rod")
    a2.set_ylabel("FCT at EOL [K]")
    title = "validation rods"
    if vsum is not None and len(vsum):
        t = vsum.set_index("method")
        title += "\n" + ", ".join(
            fr"{'S' if m == 'SC' else 'H'}: $\chi^2_E$={t.loc[m, 'chi2_E']:.2f}, $\chi^2_{{tot}}$={t.loc[m, 'chi2_tot']:.2f}"
            for m in ("SC", "HC") if m in t.index)
    a2.set_title(title, fontsize=11)
    h, lab = a1.get_legend_handles_labels()
    h.append(Line2D([0], [0], marker="D", color=C_EXP, lw=0))
    lab.append(r"validation measurement $\pm1.96\,\sigma_{exp}$")
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=4, frameon=False)
    fig.suptitle(f"{CASE_TITLE[name]}: posterior predictive (GP variance included)")
    fig.tight_layout()
    _save(fig, out / f"predictive_{name}")


# --------------------------------------------------------------------------- #
#  Tables
# --------------------------------------------------------------------------- #
def convergence_rows(c: dict, name: str) -> list:
    """Verdicts from the latest step-06 tables of the case."""
    rows = []
    d = Path(c["cfg"]["project"]["output_dir"])
    d = d if d.is_absolute() else Path(c["cfg"]["_config_dir"]) / d
    for m, f in c["files"].items():
        conv = d / "diagnostics" / f"{f.stem}_convergence.csv"
        if not conv.exists():
            continue
        t = pd.read_csv(conv)
        md = c["res"][m]["metadata"]
        for _, r in t[t["block"] != "X"].iterrows():
            rows.append({"case": name, "method": m, "block": r["block"], "parameter": r["parameter"],
                         "nsteps": int(md["nsteps"]), "rhat": r["rhat"], "ess_bulk": r["ess_bulk"],
                         "ess_tail": r["ess_tail"], "tau": r["tau"], "n_over_tau": r["n_over_tau"],
                         "acceptance": r["acceptance"], "pass": r["pass"],
                         "wall_clock_min": float(np.max(np.atleast_1d(md["wall_clock_s"]))) / 60})
        x = t[t["block"] == "X"]
        if len(x):
            rows.append({"case": name, "method": m, "block": "X (all rods)", "parameter": "worst",
                         "nsteps": int(md["nsteps"]), "rhat": x["rhat"].max(), "ess_bulk": x["ess_bulk"].min(),
                         "ess_tail": x["ess_tail"].min(), "tau": x["tau"].max(),
                         "n_over_tau": x["n_over_tau"].min(),
                         "acceptance": f"{x['acceptance'].min():.2f}-{x['acceptance'].max():.2f}",
                         "pass": bool(x["acceptance_ok"].all()), "wall_clock_min": np.nan})
    return rows


def fig_prior_sensitivity(cu: dict, cn: dict, out: Path):
    """Epistemic case: posteriors with the uniform prior (solid) and with the literature
    normal priors (dashed), standard and hierarchical; priors in grey; true value as rug."""
    from scipy.stats import truncnorm
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, param in zip(axes, ORDER):
        j = cu["post"]["SC"].names.index(param)
        lo, hi = cu["post"]["SC"].lower_phys[j], cu["post"]["SC"].upper_phys[j]
        ax.hlines(1.0 / (hi - lo), lo, hi, color=C_PRIOR, lw=1.8, ls="-", zorder=2)
        pr = cn["cfg"]["parameters"][param]["prior"]
        xg = np.linspace(lo, hi, 1000)
        m_, s_ = float(pr["mean"]), float(pr["std"])
        ax.plot(xg, truncnorm.pdf(xg, (lo - m_) / s_, (hi - m_) / s_, loc=m_, scale=s_),
                color=C_PRIOR, lw=1.8, ls="--", zorder=2)
        peak = 0
        for c, ls in ((cu, "-"), (cn, "--")):
            for m, col in (("SC", C_SC), ("HC", C_HC)):
                xx, yy = bounded_kde(samples_of(c, m)[:, j], lo, hi)
                ax.plot(xx, yy, color=col, lw=2.2, ls=ls, zorder=4)
                peak = max(peak, yy.max())
        ax.vlines(float(cu["meta"]["true_A"][param]), 0, RUG_H, color=C_TRUE, lw=RUG_LW, zorder=6,
                  transform=ax.get_xaxis_transform())
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, 1.12 * peak)
        ax.set_xlabel(LABEL.get(param, param) + " (multiplier)")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("PDF")
    handles = [Line2D([0], [0], color=C_PRIOR, lw=1.8), Line2D([0], [0], color=C_PRIOR, lw=1.8, ls="--"),
               Line2D([0], [0], color=C_SC, lw=2.2), Line2D([0], [0], color=C_HC, lw=2.2),
               Line2D([0], [0], color="k", lw=2.2), Line2D([0], [0], color="k", lw=2.2, ls="--"),
               Line2D([0], [0], marker="|", ms=14, mew=RUG_LW, lw=0, color=C_TRUE)]
    labels = ["uniform prior (GP range)", "literature prior (normal, truncated)",
              "standard calibration: posterior", "hierarchical calibration: population (new rod)",
              "solid: with the uniform prior", "dashed: with the literature prior", "true value"]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=3, frameon=False)
    fig.suptitle("Epistemic uncertainty: sensitivity to the prior")
    fig.tight_layout()
    _save(fig, out / "prior_sensitivity_epistemic")


def _save(fig, stem: Path, dpi: int = 200):
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_all(configs: dict, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    post_rows, val_rows, conv_rows = [], [], []
    loaded = {}
    for name, config in configs.items():
        print(f"  figures: {name}")
        try:
            c = load_case(Path(config))
        except FileNotFoundError as exc:          # case not run yet
            print(f"    skipped: {exc}")
            continue
        d = Path(c["cfg"]["project"]["output_dir"])
        d = d if d.is_absolute() else Path(c["cfg"]["_config_dir"]) / d
        vfile = d / "validation" / "validation_surrogate_summary.csv"
        vsum = pd.read_csv(vfile) if vfile.exists() else None
        post_rows += fig_parameters(c, name, out)
        fig_traces(c, name, out)
        fig_predictive(c, name, out, vsum)
        if vsum is not None:
            val_rows += [dict(case=name, **r) for r in vsum.drop(columns=["file"]).to_dict("records")]
        conv_rows += convergence_rows(c, name)
        loaded[name] = c
    if "epistemic" in loaded and "epistemic_prior" in loaded:
        fig_prior_sensitivity(loaded["epistemic"], loaded["epistemic_prior"], out)
    pd.DataFrame(post_rows).to_csv(out / "summary_posterior.csv", index=False)
    pd.DataFrame(val_rows).to_csv(out / "summary_validation.csv", index=False)
    pd.DataFrame(conv_rows).to_csv(out / "summary_convergence.csv", index=False)
    pd.set_option("display.width", 220)
    if val_rows:
        print("\nValidation (6 rods, ideal chi2 = 1):")
        print(pd.DataFrame(val_rows)[["case", "method", "chi2_E", "chi2_tot", "coverage", "rmse"]]
              .to_string(index=False, float_format="%.3f"))
    print("\nPosterior vs truth:")
    print(pd.DataFrame(post_rows)[["case", "method", "parameter", "mean", "std", "q2.5", "q97.5",
                                   "true_value_or_population_mean", "true_in_CI95", "rods_in_CI95_%"]]
          .to_string(index=False, float_format="%.4f"))
    print(f"\nFigures and tables in {out}")


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
    cfgs = {n: root / "_configs" / f"config_{n}.yaml" for n in CASES}
    make_all({n: p for n, p in cfgs.items() if p.exists()}, root / "figures")

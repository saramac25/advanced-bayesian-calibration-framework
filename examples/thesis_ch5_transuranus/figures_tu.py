"""Figures, tables and appendix files of the TRANSURANUS / Halden (IFPE) case study
(Chapter 5, Section 5.4), from the results of run_thesis_tu.py.

    python figures_tu.py                         # results/ (after the stages correlation ... calibration)
    python figures_tu.py --results results_facsimile --examples <case1> <case2>

Outputs in <results>/figures_tu/ (PDF, plus PNG for a quick look):

  Main text
    tu_spearman_summary      variance-weighted Spearman rho^2 of the 5 multipliers, every case
                             (sum over the PCs of the share of variance x rho^2; dot = influential)
    tu_gp_parity             GP on the test runs, temperatures of all cases (a), test RMSE of each
                             case against the measurement uncertainty (b)
    tu_population            prior, standard posterior, hierarchical mu and population predictive
    tu_case_parameters       case-level posterior X[e] of the calibration cases (95% interval),
                             with mu, the population and the cases whose likelihood maximum is at a
                             bound of the GP training range
    tu_examples              prediction and C/E (prior, standard, hierarchical bands) of the --examples cases (default: 2 validation cases)
  Appendix
    appendix_A/tu_pca_<case>     cumulative explained variance (a), Spearman rho^2 per PC (b)
    appendix_B/tu_pred_<case>    prediction (top) and C/E (bottom), standard and hierarchical
    appendix_tu_A.tex, appendix_tu_B.tex   LaTeX of the two appendices (captions with the metrics)
  Tables (CSV)
    tu_cases.csv             set, points, PCs, explained variance, GP test R2/RMSE of every case
    tu_case_metrics.csv      coverage, chi2_E, chi2_tot of every case and method
    tu_case_modes.csv        likelihood maximum of every calibration case, at a bound or not

Predictions (one band = 95% central interval of the predictive distribution):
  * standard calibration: theta from the posterior, for every case;
  * hierarchical: theta_new = mu + sigma z (population predictive), for every case (band drawn);
  * hierarchical, calibration case e, metrics only (HC_case): theta from the case-level
    posterior X[e], i.e. the fit of the case's own history.
Each theta is propagated through the GP of the case (mean + GP variance) and the PCA. The
measurement uncertainty (1%) is NOT in the bands; it is drawn as error bars / grey band
(C/E = 1 +- 1.96 x 1%). The coverage and chi2 of the captions follow step 08 (predictive
distribution including the measurement noise; chi2_E at the MAP point, i.e. theta for the
standard and X[e] for a calibration case / mu for a validation case for the hierarchical).
Prior band: the calibrated multipliers uniform over the GP training range (the prior), the
others at 1. C/E = predicted / measured temperature, computed draw by draw. Nominal: all multipliers = 1,
GP mean.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from abcf.config import case_noise, param_label, validation_case_list  # noqa: E402
from abcf.io import load_results  # noqa: E402
from abcf.postprocessing.posterior import Posterior  # noqa: E402
from abcf.validation.metrics import case_metrics, chi2_terms  # noqa: E402
from abcf.workflow import calibration_cases, latest_result, load_all  # noqa: E402

HERE = Path(__file__).resolve().parent
C_SC, C_HC, C_PRIOR, C_EXP, C_NOM, C_BOUND = "#7b3294", "#e69f00", "0.55", "#0b2c5a", "0.15", "#d7191c"
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 12,
                     "axes.titlesize": 13, "axes.labelsize": 13, "legend.fontsize": 10,
                     "axes.linewidth": 1.0})
N_DRAWS = 2000
LEVEL = 0.95


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def tex(s: str) -> str:
    return s.replace("_", r"\_")


def case_title(cid: str) -> str:
    """'123-4_5-rod-3-node-10' -> 'IFA-123-4_5 ROD3, node 10'."""
    try:
        exp, rest = cid.split("-rod-")
        rod, node = rest.split("-node-")
        node = "node-mean value" if node == "mean" else f"node {node}"
        return f"IFA-{exp} ROD{rod}, {node}"
    except ValueError:
        return cid


def lab(cfg, name):
    return param_label(cfg, name)


class Predictor:
    """theta (physical, calibrated parameters) -> temperature draws through the GP + PCA."""

    def __init__(self, setup, sset, seed=0):
        self.setup, self.sset = setup, sset
        self.rng = np.random.default_rng(seed)

    def mean(self, theta, cid):
        Xs = self.sset.scaler.transform(self.setup.full_vector(theta, cid))
        return self.sset[cid].predict_qoi(Xs)

    def draws(self, theta, cid):
        """GP mean + GP variance (diagonal in the QoI space, as step 08)."""
        sur = self.sset[cid]
        Xs = self.sset.scaler.transform(self.setup.full_vector(theta, cid))
        m, v = sur.predict(Xs)
        red = sur.reduction
        pred = red.reconstruct(m)
        sd = np.sqrt(np.maximum(v @ red.W ** 2, 0.0))
        return pred + self.rng.standard_normal(pred.shape) * sd, sd


# --------------------------------------------------------------------------- #
#  Sensitivity
# --------------------------------------------------------------------------- #
def spearman_summary(io: pd.DataFrame, cfg, cases_order, val, out):
    io = io.copy()
    io["w"] = io["variance_share"] * io["rho_squared"]
    w = io.groupby(["case", "parameter"])["w"].sum().unstack()
    infl = io.groupby(["case", "parameter"])["influential"].any().unstack()
    params = list(dict.fromkeys(io["parameter"]))
    w, infl = w.loc[cases_order, params], infl.loc[cases_order, params]
    fig, ax = plt.subplots(figsize=(1.25 * len(params) + 3.2, 0.24 * len(cases_order) + 1.6))
    im = ax.imshow(w.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    yi, xi = np.where(infl.values)
    ax.plot(xi, yi, "o", ms=3.2, color="white", mec="k", mew=0.5)
    ax.set_xticks(range(len(params)), [lab(cfg, p) for p in params], rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(cases_order)), cases_order, fontsize=7.5)
    for t, c in zip(ax.get_yticklabels(), cases_order):
        if c in val:
            t.set_fontweight("bold")
            t.set_color(C_SC)
    fig.colorbar(im, ax=ax, label=r"weighted $\rho_S^2$", shrink=0.6)
    ax.set_title("Variance-weighted Spearman " + r"$\rho_S^2$", fontsize=12)
    handles = [Line2D([], [], marker="o", ls="", color="white", mec="k", ms=5,
                      label="influential (BH-FDR, at least one PC)"),
               Line2D([], [], ls="", marker="s", color=C_SC, ms=5, label="validation case (bold)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.07 - 2.0 / len(cases_order)),
              frameon=False, fontsize=9)
    save(fig, out / "tu_spearman_summary")
    pd.concat({"weighted_rho2": w, "influential": infl}, axis=1).to_csv(out / "tu_spearman_summary.csv")
    frac = infl.mean()
    print("Influential in the fraction of cases: " + ", ".join(f"{p} {frac[p]:.0%}" for p in params))


def appendix_A(data, sset, io, cfg, cases_order, val, out, pca_var):
    d = out / "appendix_A"
    lines = ["% Appendix: PCA and Spearman screening of every case (figures_tu.py)", ""]
    for i, cid in enumerate(cases_order):
        Y = data.cases[cid].pred
        ratio = np.cumsum(PCA().fit(Y).explained_variance_ratio_)
        k = sset[cid].n_components
        sub = io[io["case"] == cid]
        params = list(dict.fromkeys(sub["parameter"]))
        piv = sub.pivot(index="parameter", columns="output", values="rho_squared").loc[params]
        sgn = sub.pivot(index="parameter", columns="output", values="spearman_rho").loc[params]
        sig = sub.pivot(index="parameter", columns="output", values="influential").loc[params]
        share = sub.drop_duplicates("output").set_index("output")["variance_share"].reindex(piv.columns)
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.0, 1.1]})
        ax = axes[0]
        n_show = min(len(ratio), max(6, k + 3))
        ax.plot(np.arange(1, n_show + 1), 100 * ratio[:n_show], "o-", color=C_EXP, ms=4)
        ax.axhline(100 * pca_var, color="k", ls="--", lw=0.8)
        ax.axvline(k, color=C_HC, ls=":", lw=1.2)
        ax.set_xlabel("number of components")
        ax.set_ylabel("cumulative explained variance [%]")
        ax.set_title(f"(a) {k} PC retained ({100 * ratio[k - 1]:.2f}%)", fontsize=11)
        ax.set_xticks(range(1, n_show + 1))
        ax = axes[1]
        im = ax.imshow(piv.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for a in range(piv.shape[0]):
            for b in range(piv.shape[1]):
                v = piv.values[a, b]
                t = f"{v:.2f} ({'+' if sgn.values[a, b] >= 0 else '-'})" + ("" if sig.values[a, b] else "\nn.i.")
                ax.text(b, a, t, ha="center", va="center", fontsize=8, color="white" if v > 0.6 else "black")
        ax.set_xticks(range(piv.shape[1]), [f"{c}\n({100 * s:.1f}%)" for c, s in zip(piv.columns, share)],
                      fontsize=9)
        ax.set_yticks(range(len(params)), [lab(cfg, p) for p in params], fontsize=9)
        ax.set_title(r"(b) Spearman $\rho_S^2$ (sign of $\rho_S$)", fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        save(fig, d / f"tu_pca_{cid}")
        setname = "validation" if cid in val else "calibration"
        lines += [r"\begin{figure}[!htbp]", r"    \centering",
                  rf"    \includegraphics[width=\textwidth]{{images/appendix_tu/appendix_A/tu_pca_{cid}.pdf}}",
                  rf"    \caption{{{tex(case_title(cid))} ({setname} case): (a) cumulative explained variance "
                  rf"of the PCA of the {Y.shape[0]} TRANSURANUS runs, with the {100 * pca_var:.0f}\% threshold "
                  rf"(dashed) and the number of retained components (dotted); (b) squared Spearman "
                  rf"correlation between the five multipliers and the retained scores (share of the "
                  rf"variance of each component in brackets); n.i.: not influential after the "
                  rf"Benjamini--Hochberg correction.}}",
                  rf"    \label{{fig:app_tu_A_{i + 1}}}", r"\end{figure}"]
        lines += [r"\clearpage", ""] if i % 2 == 1 else [""]
    (out / "appendix_tu_A.tex").write_text("\n".join(lines))


def gp_parity(data, sset, cfg_s, cases_order, noise, out):
    seed, tf = int(cfg_s["project"]["seed"]), cfg_s["surrogate"]["test_fraction"]
    sim, gp, rows = [], [], []
    rng = np.random.default_rng(0)
    for cid in cases_order:
        case = data.cases[cid]
        X = data.X(case)
        _, it = train_test_split(np.arange(X.shape[0]), test_size=tf, random_state=seed)
        p = sset[cid].predict_qoi(sset.scaler.transform(X[it]))
        y = case.pred[it]
        rows.append({"case": cid, "rmse_test_K": float(np.sqrt(np.mean((p - y) ** 2))),
                     "sigma_exp_K": float(noise * np.mean(np.abs(case.meas)))})
        k = rng.choice(y.size, size=min(y.size, 3000), replace=False)
        sim.append(y.ravel()[k])
        gp.append(p.ravel()[k])
    sim, gp = np.concatenate(sim), np.concatenate(gp)
    r = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.5]})
    ax = axes[0]
    ax.plot(sim, gp, ".", ms=1.5, alpha=0.3, color=C_EXP, rasterized=True)
    lim = [min(sim.min(), gp.min()), max(sim.max(), gp.max())]
    ax.plot(lim, lim, "k--", lw=0.8)
    r2 = 1 - np.sum((gp - sim) ** 2) / np.sum((sim - sim.mean()) ** 2)
    ax.set_xlabel("TRANSURANUS temperature [K]")
    ax.set_ylabel("GP prediction [K]")
    ax.set_title(f"(a) test runs, all cases ($R^2$ = {r2:.4f})", fontsize=11)
    ax = axes[1]
    x = np.arange(len(r))
    ax.bar(x, r["rmse_test_K"], color=C_EXP, label="GP test RMSE")
    ax.plot(x, r["sigma_exp_K"], "_", ms=10, mew=2, color=C_HC, label=r"$\sigma_{\exp}$ (1% of the mean measurement)")
    ax.set_xticks(x, r["case"], rotation=90, fontsize=7)
    ax.set_ylabel("[K]")
    ax.set_title("(b) test RMSE of each case", fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    save(fig, out / "tu_gp_parity")
    return r


# --------------------------------------------------------------------------- #
#  Posterior
# --------------------------------------------------------------------------- #
def population_figure(sc, hc, cfg, out):
    p = len(hc.names)
    pop = hc.population(20000)
    fig, axes = plt.subplots(1, p, figsize=(6.2 * p, 4.2), squeeze=False)
    for j, n in enumerate(hc.names):
        ax = axes[0, j]
        lo, hi = hc.lower_phys[j], hc.upper_phys[j]
        bins = np.linspace(lo, hi, 70)
        ax.hlines(1 / (hi - lo), lo, hi, color=C_PRIOR, ls="--", lw=1.5, label="prior (GP training range)")
        ax.hist(pop[:, j], bins=bins, density=True, histtype="step", lw=2, color=C_HC,
                label=r"hierarchical: population $\theta_{new}$")
        ax.hist(hc.mu[:, j], bins=bins, density=True, alpha=0.3, color=C_HC, label=r"hierarchical: $\mu$")
        w = np.diff(np.percentile(sc.theta[:, j], [2.5, 97.5]))[0] / np.median(sc.theta[:, j])
        ax.axvline(np.median(sc.theta[:, j]), color=C_SC, lw=2,
                   label=f"standard: posterior (95% width {100 * w:.2g}%)" if j == 0 else None)
        ax.axvline(1.0, color=C_NOM, ls=":", lw=1.2, label="nominal")
        ax.set_xlabel(lab(cfg, n))
        ax.set_ylabel("density")
        ax.set_xlim(lo, hi)
    h, l_ = axes[0, 0].get_legend_handles_labels()
    if p > 1:
        for j in range(1, p):
            w = np.diff(np.percentile(sc.theta[:, j], [2.5, 97.5]))[0] / np.median(sc.theta[:, j])
            l_ = [x.replace("width", f"width {lab(cfg, hc.names[0])}:").split(" (")[0] + " (95% width < "
                  + f"{100 * max(w, np.diff(np.percentile(sc.theta[:, 0], [2.5, 97.5]))[0] / np.median(sc.theta[:, 0])):.2g}%)"
                  if x.startswith("standard") else x for x in l_]
    fig.legend(h, l_, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False)
    fig.tight_layout()
    save(fig, out / "tu_population")


def case_parameters(sc, hc, modes_df, cfg, out):
    p = len(hc.names)
    cases = hc.case_ids
    pop = hc.population(20000)
    bound = set(modes_df.loc[modes_df["at_bound"], "case"])
    fig, axes = plt.subplots(1, p, figsize=(6.0 * p, 0.27 * len(cases) + 2.0), squeeze=False, sharey=True)
    y = np.arange(len(cases))
    for j, n in enumerate(hc.names):
        ax = axes[0, j]
        q = np.percentile(hc.X[:, :, j], [2.5, 50, 97.5], axis=0)
        qp = np.percentile(pop[:, j], [2.5, 97.5])
        qm = np.percentile(hc.mu[:, j], [2.5, 97.5])
        ax.axvspan(*qp, color=C_HC, alpha=0.12, label="population 95%")
        ax.axvspan(*qm, color=C_HC, alpha=0.35, label=r"$\mu$ 95%")
        ax.axvline(np.median(sc.theta[:, j]), color=C_SC, lw=1.8, label="standard")
        for e, cid in enumerate(cases):
            col = C_BOUND if cid in bound else "k"
            ax.errorbar(q[1, e], y[e], xerr=[[q[1, e] - q[0, e]], [q[2, e] - q[1, e]]], fmt="o", ms=3.5,
                        color=col, capsize=2, lw=1)
        ax.axvline(hc.lower_phys[j], color=C_PRIOR, ls="--", lw=1)
        ax.axvline(hc.upper_phys[j], color=C_PRIOR, ls="--", lw=1)
        ax.set_xlabel(lab(cfg, n))
        ax.set_yticks(y, cases, fontsize=7.5)
        ax.invert_yaxis() if j == 0 else None
    handles = [Patch(color=C_HC, alpha=0.12, label="population, 95%"),
               Patch(color=C_HC, alpha=0.35, label=r"$\mu$, 95%"),
               Line2D([], [], color=C_SC, lw=1.8, label="standard"),
               Line2D([], [], color="k", marker="o", ls="", ms=4, label=r"case $X_e$: median, 95%"),
               Line2D([], [], color=C_BOUND, marker="o", ls="", ms=4, label="likelihood maximum at a bound"),
               Line2D([], [], color=C_PRIOR, ls="--", label="GP training range")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False)
    fig.tight_layout()
    save(fig, out / "tu_case_parameters")


def compute_modes(cfg, data, sset, setup, cases, out):
    from abcf.calibration.hierarchical.runner import case_modes
    from abcf.calibration.likelihood import build_likelihoods
    idx = setup.calibrated_idx
    lo, hi = sset.scaler.lower[idx], sset.scaler.upper[idx]
    m, ll = case_modes(build_likelihoods(cfg, data, sset, setup, cases), lo, hi, seed=2026)
    at = np.any((m <= lo + 1e-6) | (m >= hi - 1e-6), axis=1)
    phys = m * sset.scaler.std[idx] + sset.scaler.mean[idx]
    df = pd.DataFrame(phys, columns=setup.calibrated_names)
    df.insert(0, "case", cases)
    df["loglik_max"] = ll
    df["at_bound"] = at
    df.to_csv(out / "tu_case_modes.csv", index=False)
    print(f"{int(at.sum())} of {len(cases)} calibration cases with the likelihood maximum at a bound: "
          + ", ".join(df.loc[at, "case"]))
    return df


# --------------------------------------------------------------------------- #
#  Predictions
# --------------------------------------------------------------------------- #
def case_predictions(cid, is_val, sc, hc, pr, cfg, data, seed):
    """Draws, bands and metrics of one case for the two methods."""
    case = data.cases[cid]
    y = case.meas
    r = case_noise(cfg, cid)
    res = {"nominal": pr.mean(np.ones(len(sc.names)), cid)[0]}
    # prior predictive: calibrated multipliers uniform over the GP training range (the prior)
    rng = np.random.default_rng(seed + 1)
    th0 = rng.uniform(hc.lower_phys, hc.upper_phys, size=(N_DRAWS, len(hc.names)))
    d0, _ = pr.draws(th0, cid)
    lv = [100 * (1 - LEVEL) / 2, 100 * (1 + LEVEL) / 2]
    res["prior"] = {"band": np.percentile(d0, lv, axis=0), "ce_band": np.percentile(d0 / y, lv, axis=0),
                    "coverage": case_metrics(d0, y, r, LEVEL, seed)["coverage"]}
    # SC: posterior theta; HC: population theta_new = mu + sigma z (the band drawn, every case);
    # HC_case (calibration cases only, metrics only): case-level posterior X[e]
    methods = [("SC", sc), ("HC", hc)] + ([] if is_val else [("HC_case", hc)])
    for m, post in methods:
        if m == "SC":
            theta, theta_map = post.draws(N_DRAWS, seed), post.map_point()
        elif m == "HC":
            theta, theta_map = post.population(N_DRAWS, seed), post.map_point()
        else:
            e = post.case_ids.index(cid)
            theta = post.case_draws(e, N_DRAWS, seed)
            theta_map = post.X[int(np.nanargmax(post._logp)), e] if post._logp is not None \
                else np.median(post.X[:, e], axis=0)
        d, _ = pr.draws(theta, cid)
        sur = pr.sset[cid]
        Xs = pr.sset.scaler.transform(pr.setup.full_vector(theta_map, cid))
        zm, vm = sur.predict(Xs)
        p_fit = sur.reduction.reconstruct(zm)[0]
        sig_gp = np.sqrt(np.maximum((vm @ sur.reduction.W ** 2)[0], 0.0))
        sig_post = pr.mean(theta, cid).std(axis=0, ddof=1)
        t = chi2_terms(y, p_fit, sig_post, sig_gp, r * np.abs(y))
        cm = case_metrics(d, y, r, LEVEL, seed)
        res[m] = {"band": np.percentile(d, lv, axis=0), "ce_band": np.percentile(d / y, lv, axis=0),
                  "coverage": cm["coverage"],
                  "chi2_E": float(t["chi2_E_term"].mean()), "chi2_tot": float(t["chi2_tot_term"].mean())}
    return res


C_MEAS = "#d7191c"     # measurements in red, distinct from the nominal prediction (black)


def _break_gaps(t, *arrays, frac=0.01):
    """Insert NaN where the time step exceeds `frac` of the time span (shutdowns between
    cycles), so that lines and bands are not drawn across periods without data."""
    t = np.asarray(t, float)
    dt = np.diff(t)
    pos = dt[dt > 0]
    thr = max(frac * (t.max() - t.min()), 20 * (np.median(pos) if len(pos) else 0.0))
    gap = np.where(dt > thr)[0] + 1
    out = [np.insert(t, gap, np.nan)]
    for a in arrays:
        out.append(np.insert(np.asarray(a, float), gap, np.nan))
    return out


def plot_case(ax_t, ax_ce, cid, case, res, r, time_label, legend=True):
    t, y = case.tim, case.meas
    o = np.argsort(t, kind="stable")
    ts, ys = t[o], y[o]
    # prior predictive: light grey filled band, below everything
    for ax, key in ((ax_t, "band"), (ax_ce, "ce_band")):
        tt, lo, hi = _break_gaps(ts, res["prior"][key][0][o], res["prior"][key][1][o])
        ax.fill_between(tt, lo, hi, color="0.85", lw=0, zorder=1,
                        label="prior: 95% band" if key == "band" else None)
    # hierarchical (wider), then standard on top with its edges drawn
    for m, col, name, a in (("HC", C_HC, "hierarchical", 0.45), ("SC", C_SC, "standard", 0.6)):
        for ax, key in ((ax_t, "band"), (ax_ce, "ce_band")):
            tt, lo, hi = _break_gaps(ts, res[m][key][0][o], res[m][key][1][o])
            ax.fill_between(tt, lo, hi, color=col, alpha=a, lw=0, zorder=2,
                            label=f"{name}: 95% band" if key == "band" else None)
            if m == "SC":
                ax.plot(tt, lo, color=col, lw=0.4, zorder=3)
                ax.plot(tt, hi, color=col, lw=0.4, zorder=3)
    tt, nom, ce_nom = _break_gaps(ts, res["nominal"][o], res["nominal"][o] / ys)
    ax_t.plot(tt, nom, color="k", lw=0.7, zorder=4, label="nominal")
    ax_ce.plot(tt, ce_nom, color="k", lw=0.5, zorder=4)
    ax_t.errorbar(ts, ys, yerr=1.96 * r * np.abs(ys), fmt="o", ms=1.2, mew=0, elinewidth=0.25,
                  color=C_MEAS, ecolor=C_MEAS, alpha=0.7, zorder=5,
                  label=r"measured $\pm 1.96\,\sigma_{\exp}$")
    for v in (1 - 1.96 * r, 1 + 1.96 * r):
        ax_ce.axhline(v, color=C_MEAS, ls="--", lw=0.8, zorder=5)
    ax_ce.plot([], [], color=C_MEAS, ls="--", lw=0.8, label=r"C/E $= 1 \pm 1.96\,\sigma_{\exp}$")
    ax_ce.axhline(1.0, color=C_MEAS, lw=0.6, zorder=5)
    ax_t.set_ylabel("FCT [K]")
    ax_ce.set_ylabel("C/E")
    ax_ce.set_xlabel(time_label)
    h1, l1 = ax_t.get_legend_handles_labels()
    h2, l2 = ax_ce.get_legend_handles_labels()
    return h1 + h2, l1 + l2


def legend_below(fig, handles, labels):
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3,
               frameon=False, fontsize=9, markerscale=4)


def metrics_text(res, pct="\\%"):
    """Coverage and chi2_tot of one case (LaTeX by default; pct="%" for the terminal)."""
    return (f"Coverage: prior {100 * res['prior']['coverage']:.0f}{pct}; "
            f"standard {100 * res['SC']['coverage']:.0f}{pct}, "
            f"$\\chi^2_{{tot}}$ = {res['SC']['chi2_tot']:.3g}; "
            f"hierarchical {100 * res['HC']['coverage']:.0f}{pct}, "
            f"$\\chi^2_{{tot}}$ = {res['HC']['chi2_tot']:.3g}"
            + (f" (case-level posterior $X_e$: {100 * res['HC_case']['coverage']:.0f}{pct}, "
               f"$\\chi^2_{{tot}}$ = {res['HC_case']['chi2_tot']:.3g})" if "HC_case" in res else ""))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results", help="results folder of run_thesis_tu.py")
    ap.add_argument("--examples", nargs="*", default=None,
                    help="cases of the main-text figure (default: first two validation cases)")
    ap.add_argument("--time-label", default="Time [d]", help="x label of the time axis")
    ap.add_argument("--skip-modes", action="store_true", help="do not recompute the likelihood maxima")
    args = ap.parse_args()

    results = (HERE / args.results).resolve()
    cfg_path = results / "_configs" / "config_tu.yaml"
    cfg_s_path = results / "_configs" / "config_surrogates.yaml"
    if not cfg_path.exists():
        sys.exit(f"{cfg_path} not found: run run_thesis_tu.py first.")
    cfg, data, setup, sset = load_all(cfg_path)
    from abcf.config import load_config
    cfg_s = load_config(cfg_s_path)
    out = results / "figures_tu"
    out.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["project"]["seed"])

    val = validation_case_list(cfg)
    cal = calibration_cases(cfg, data)
    order = cal + val
    io = pd.read_csv(results / "common" / "correlation" / "input_output_per_case.csv")
    io["case"] = io["case"].astype(str)
    noise = float(cfg["measurement_noise"]["relative"])

    print("=== sensitivity")
    spearman_summary(io, cfg, order, val, out)
    appendix_A(data, sset, io, cfg, order, val, out, cfg_s["surrogate"]["pca_variance"])
    print("=== GPs")
    gpr = gp_parity(data, sset, cfg_s, order, noise, out)
    summ = pd.read_csv(results / "common" / "surrogates" / "surrogate_summary.csv")
    summ["case"] = summ["case"].astype(str)
    tab = pd.DataFrame({"case": order, "set": ["validation" if c in val else "calibration" for c in order],
                        "n_points": [data.cases[c].n_t for c in order]})
    tab = tab.merge(summ[["case", "n_components", "explained_variance", "r2_test"]], on="case", how="left")
    tab = tab.merge(gpr, on="case", how="left")
    tab.to_csv(out / "tu_cases.csv", index=False)

    print("=== posterior")
    sc = Posterior(load_results(latest_result(cfg, "SC")))
    hc = Posterior(load_results(latest_result(cfg, "HC")))
    population_figure(sc, hc, cfg, out)
    mpath = out / "tu_case_modes.csv"
    if args.skip_modes and mpath.exists():
        modes = pd.read_csv(mpath)
    else:
        modes = compute_modes(cfg, data, sset, setup, hc.case_ids, out)
    case_parameters(sc, hc, modes, cfg, out)

    print("=== predictions")
    pr = Predictor(setup, sset, seed)
    allres, rows = {}, []
    for cid in order:
        res = case_predictions(cid, cid in val, sc, hc, pr, cfg, data, seed)
        allres[cid] = res
        for m in [k for k in ("SC", "HC", "HC_case") if k in res]:
            rows.append({"case": cid, "set": "validation" if cid in val else "calibration", "method": m,
                         "n_points": data.cases[cid].n_t, "coverage": res[m]["coverage"],
                         "chi2_E": res[m]["chi2_E"], "chi2_tot": res[m]["chi2_tot"]})
    met = pd.DataFrame(rows)
    met.to_csv(out / "tu_case_metrics.csv", index=False)
    g = met.groupby(["set", "method"]).agg(cases=("case", "count"), coverage_mean=("coverage", "mean"),
                                          chi2_tot_median=("chi2_tot", "median"))
    print(g.round(3).to_string())

    lines = ["% Appendix: predictions of every case (figures_tu.py)", ""]
    for i, cid in enumerate(order):
        case = data.cases[cid]
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True, gridspec_kw={"height_ratios": [1.6, 1]})
        h, l_ = plot_case(a1, a2, cid, case, allres[cid], case_noise(cfg, cid), args.time_label)
        a1.set_title(case_title(cid), fontsize=12)
        fig.tight_layout()
        legend_below(fig, h, l_)
        save(fig, out / "appendix_B" / f"tu_pred_{cid}")
        setname = "validation case" if cid in val else "calibration case"
        lines += [r"\begin{figure}[!htbp]", r"    \centering",
                  rf"    \includegraphics[width=0.9\textwidth]{{images/appendix_tu/appendix_B/tu_pred_{cid}.pdf}}",
                  rf"    \caption{{{tex(case_title(cid))} ({setname}): fuel centreline temperature (top) and "
                  rf"C/E ratio (bottom); 95\% predictive bands of the prior (grey), of the standard "
                  rf"and of the hierarchical calibration (population predictive), nominal prediction (black) and "
                  rf"measurements (red). "
                  rf"{metrics_text(allres[cid])}.}}",
                  rf"    \label{{fig:app_tu_B_{i + 1}}}", r"\end{figure}"]
        lines += [r"\clearpage", ""] if i % 2 == 1 else [""]
    (out / "appendix_tu_B.tex").write_text("\n".join(lines))

    ex = args.examples or val[:2]
    fig, axes = plt.subplots(2, len(ex), figsize=(7.2 * len(ex), 6.4), sharex="col", squeeze=False,
                             gridspec_kw={"height_ratios": [1.6, 1]})
    for j, cid in enumerate(ex):
        h, l_ = plot_case(axes[0, j], axes[1, j], cid, data.cases[cid], allres[cid], case_noise(cfg, cid),
                          args.time_label)
        kind = "validation" if cid in val else "calibration"
        axes[0, j].set_title(f"({'abcd'[j]}) {case_title(cid)} ({kind})", fontsize=11)
    fig.tight_layout()
    legend_below(fig, h, l_)
    save(fig, out / "tu_examples")
    print("Examples: " + "; ".join(f"{c}: {metrics_text(allres[c], '%')}" for c in ex))
    print(f"\nFigures and tables in {out}")


if __name__ == "__main__":
    main()

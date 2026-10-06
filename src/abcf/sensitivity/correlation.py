"""Correlation analysis on the solver runs used to train the GPs.

1. Input-output: Spearman rank correlation between each parameter and
   each output of a case (the QoI for a scalar QoI, the PCA scores for a
   time-dependent QoI), tested for rho = 0 and corrected for multiple
   tests within the case with the Benjamini-Hochberg false discovery rate
   (Benjamini & Hochberg, 1995, J. R. Stat. Soc. B 57(1), 289-300,
   doi:10.1111/j.2517-6161.1995.tb02031.x). A parameter is "influential"
   in a case if, for at least one output, the correlation is significant
   AND |rho| >= effect_size_threshold. Pearson's r is reported as a
   non-linearity diagnostic only.
   Across cases: fraction of cases where the parameter is influential and
   mean |rho| through the Fisher z-transform (weights n - 3).

2. Output-output: Spearman correlation between the outputs of every pair
   of cases on their common solver runs (same PC index for PCA scores;
   |rho| is used because the sign of a principal component is
   arbitrary), FDR-corrected over all pairs. Pairs with
   |rho| >= redundancy_threshold are "redundant"; connected groups of
   redundant cases are reported, with one representative each, to help
   choose the calibration and validation cases.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from ..io import InputData
from ..surrogate.gp import OutputReduction


def case_outputs(data: InputData, pca_variance: float) -> dict:
    """case_id -> DataFrame of outputs (index eval_id): QoI or PCA scores."""
    out = {}
    for cid, case in data.cases.items():
        red = OutputReduction(case.pred, pca_variance)
        Z = red.project(case.pred)
        cols = ["QoI"] if case.is_scalar else [f"PC{k + 1}" for k in range(Z.shape[1])]
        df = pd.DataFrame(Z, index=case.eval_ids, columns=cols)
        df.attrs["explained_variance"] = float(np.sum(red.explained_variance_ratio))
        out[cid] = df
    return out


def input_output(data: InputData, outputs: dict, cfg_c: dict) -> pd.DataFrame:
    """Long table: one row per (case, parameter, output)."""
    rows = []
    for cid, Y in outputs.items():
        X = data.inputs.loc[Y.index]
        block = []
        for p in X.columns:
            for o in Y.columns:
                rho, pval = stats.spearmanr(X[p].values, Y[o].values)
                r, _ = stats.pearsonr(X[p].values, Y[o].values)
                block.append({"case": cid, "parameter": p, "output": o, "n_samples": len(Y),
                              "spearman_rho": rho, "p_value": pval, "pearson_r": r})
        pvals = np.array([b["p_value"] for b in block])
        reject, p_corr, _, _ = multipletests(pvals, alpha=cfg_c["alpha"], method=cfg_c["fdr_method"])
        for b, rej, pc in zip(block, reject, p_corr):
            b["p_value_corrected"] = pc
            b["significant"] = bool(rej)
            b["influential"] = bool(rej and abs(b["spearman_rho"]) >= cfg_c["effect_size_threshold"])
            b["nonlinearity_flag"] = bool(abs(b["spearman_rho"]) - abs(b["pearson_r"]) > 0.2)
        rows.extend(block)
    return pd.DataFrame(rows)


def input_output_summary(io_table: pd.DataFrame) -> pd.DataFrame:
    """Per parameter, across cases: fraction of influential cases and Fisher-z mean |rho|."""
    per_case = (io_table.assign(abs_rho=io_table["spearman_rho"].abs())
                .groupby(["parameter", "case"])
                .agg(max_abs_rho=("abs_rho", "max"), n_samples=("n_samples", "first"),
                     influential=("influential", "any"))
                .reset_index())

    def agg(g):
        z = np.arctanh(g["max_abs_rho"].clip(upper=0.999))
        w = (g["n_samples"] - 3).clip(lower=1)
        return pd.Series({"fraction_cases_influential": g["influential"].mean(),
                          "influential_in_all_cases": bool(g["influential"].all()),
                          "mean_abs_rho_fisher_z": float(np.tanh(np.average(z, weights=w))),
                          "min_abs_rho": g["max_abs_rho"].min(), "n_cases": len(g)})

    return (per_case.groupby("parameter").apply(agg, include_groups=False)
            .sort_values("fraction_cases_influential", ascending=False))


def output_output(outputs: dict, cfg_c: dict) -> pd.DataFrame:
    """Pairwise Spearman correlation between cases, per shared output index."""
    rows = []
    for a, b in itertools.combinations(outputs, 2):
        ya, yb = outputs[a], outputs[b]
        shared = np.intersect1d(ya.index, yb.index)
        for k in range(min(ya.shape[1], yb.shape[1])):
            rho, pval = stats.spearmanr(ya.iloc[:, k].loc[shared], yb.iloc[:, k].loc[shared])
            rows.append({"case_A": a, "case_B": b, "output_index": k, "n_shared": len(shared),
                         "spearman_rho": rho, "abs_rho": abs(rho), "p_value": pval})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    reject, p_corr, _, _ = multipletests(df["p_value"], alpha=cfg_c["alpha"], method=cfg_c["fdr_method"])
    df["p_value_corrected"] = p_corr
    df["significant"] = reject
    df["redundant"] = reject & (df["abs_rho"] >= cfg_c["redundancy_threshold"])
    return df


def redundancy_groups(oo_table: pd.DataFrame, case_ids: list, outputs: dict) -> pd.DataFrame:
    """Groups of cases connected by redundant pairs (first output index only).

    Representative of a group: the case whose outputs retain the most
    variance (first case for scalar QoIs). Suggestion only: the user
    decides the calibration/validation split in the config.
    """
    parent = {c: c for c in case_ids}

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    first = oo_table[(oo_table["output_index"] == 0) & oo_table["redundant"]] if len(oo_table) else oo_table
    for _, row in first.iterrows():
        parent[find(row["case_A"])] = find(row["case_B"])
    groups = {}
    for c in case_ids:
        groups.setdefault(find(c), []).append(c)

    rows = []
    for g_id, members in enumerate(groups.values(), 1):
        rep = max(members, key=lambda c: outputs[c].attrs.get("explained_variance", 1.0))
        for c in members:
            rows.append({"group": g_id, "case": c, "group_size": len(members),
                         "representative": c == rep,
                         "suggested_set": "calibration" if c == rep else "validation"})
    return pd.DataFrame(rows)

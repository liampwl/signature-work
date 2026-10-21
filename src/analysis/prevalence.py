"""
Administration comparisons at the *document* level.

v1 compared shares of paragraphs (really pages) and permuted administration
labels across paragraphs. Paragraphs from one document are not independent,
so a single long report could make a topic look like a significant shift.
Here every document counts once:

* doc_share[d, t]  = share of document d's chunks assigned to topic t
* prevalence[a, t] = mean of doc_share over documents in administration a
* permutation test shuffles administration labels across *documents*
  (optionally within document-type strata), with Benjamini-Hochberg FDR
* regression controls for document type and country when metadata exist
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
from src.utils import get_logger

logger = get_logger(__name__)


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted q-values."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1)
    return out


def prevalence_table(shares: np.ndarray, docs: pd.DataFrame, chunk_topics: np.ndarray,
                     chunk_admin: np.ndarray) -> pd.DataFrame:
    """Doc-weighted prevalence (primary) next to v1-style chunk-weighted prevalence (for comparison)."""
    is_b = (docs["administration"] == "Biden").to_numpy()
    k = shares.shape[1]
    out = pd.DataFrame({
        "topic": range(k),
        "trump_doc_weighted": shares[~is_b].mean(axis=0),
        "biden_doc_weighted": shares[is_b].mean(axis=0),
    })
    out["diff_biden_minus_trump"] = out["biden_doc_weighted"] - out["trump_doc_weighted"]
    for admin, col in [("Trump", "trump_chunk_weighted"), ("Biden", "biden_chunk_weighted")]:
        m = chunk_admin == admin
        out[col] = np.bincount(chunk_topics[m], minlength=k) / max(m.sum(), 1)
    out["n_docs_trump_any"] = (shares[~is_b] > 0).sum(axis=0)
    out["n_docs_biden_any"] = (shares[is_b] > 0).sum(axis=0)
    return out


def permutation_test(shares: np.ndarray, is_biden: np.ndarray, strata: np.ndarray | None = None,
                     n_perm: int = cfg.N_PERMUTATIONS, seed: int = cfg.RANDOM_SEED) -> tuple[np.ndarray, np.ndarray]:
    """
    Two-sided document-level permutation test of (mean Biden share - mean Trump share).
    If `strata` is given, labels are shuffled only within strata (e.g. document type),
    which holds the document-type mix of each administration fixed.
    Returns (observed_diff, p_values).
    """
    rng = np.random.default_rng(seed)
    is_biden = is_biden.astype(bool)
    n_b, n_t = is_biden.sum(), (~is_biden).sum()

    def diff(lab: np.ndarray) -> np.ndarray:
        return shares[lab].mean(axis=0) - shares[~lab].mean(axis=0)

    obs = diff(is_biden)
    exceed = np.zeros_like(obs)
    if strata is None:
        strata = np.zeros(len(is_biden), dtype=int)
    groups = [np.where(strata == s)[0] for s in np.unique(strata)]

    batch = 500
    done = 0
    while done < n_perm:
        b = min(batch, n_perm - done)
        P = np.empty((b, len(is_biden)), dtype=bool)
        for i in range(b):
            lab = is_biden.copy()
            for g in groups:
                lab[g] = lab[g][rng.permutation(len(g))]
            P[i] = lab
        Pb = P.astype(np.float64)
        diffs = (Pb @ shares) / n_b - ((1 - Pb) @ shares) / n_t
        exceed += (np.abs(diffs) >= np.abs(obs) - 1e-12).sum(axis=0)
        done += b
    p = (exceed + 1) / (n_perm + 1)
    return obs, p


def bootstrap_ci(shares: np.ndarray, is_biden: np.ndarray, n_boot: int = 2000,
                 seed: int = cfg.RANDOM_SEED) -> tuple[np.ndarray, np.ndarray]:
    """95% percentile CI for the Biden-Trump difference, resampling documents within each period."""
    rng = np.random.default_rng(seed)
    b, t = shares[is_biden], shares[~is_biden]
    diffs = np.empty((n_boot, shares.shape[1]))
    for i in range(n_boot):
        diffs[i] = b[rng.integers(0, len(b), len(b))].mean(0) - t[rng.integers(0, len(t), len(t))].mean(0)
    return np.quantile(diffs, 0.025, axis=0), np.quantile(diffs, 0.975, axis=0)


def covariate_regressions(shares: np.ndarray, docs: pd.DataFrame, trend: bool = False) -> pd.DataFrame | None:
    """
    Per topic: OLS of document share on Biden + document type + country (HC3 SEs).
    Linear probability on shares is crude but transparent; the point is whether
    the administration coefficient survives the controls.

    trend=True also adds a linear time trend and a COVID-period indicator
    (documents dated on/after 2020-03-11). The Biden coefficient then measures
    a jump at the inauguration over and above a steady trend and the pandemic.
    Administration and time are confounded in this design, so this is the
    stricter test: a topic that was already rising or falling before 2021, or
    that moved with COVID (which began under Trump), loses its "Biden effect".
    """
    import statsmodels.formula.api as smf

    has_type = docs["doc_type"].nunique() > 1
    has_country = docs["country"].nunique() > 1
    if not (has_type or has_country):
        logger.info("No document-type/country metadata; skipping covariate regressions.")
        return None

    df = docs[["administration", "doc_type", "country", "date"]].copy()
    df["biden"] = (df["administration"] == "Biden").astype(int)
    years = df["date"].dt.year + (df["date"].dt.dayofyear - 1) / 365.25
    df["year_c"] = years - 2021.05
    df["covid"] = (df["date"] >= pd.Timestamp("2020-03-11")).astype(int)
    top = df["country"].value_counts().head(cfg.TOP_COUNTRIES).index
    df["country_c"] = np.where(df["country"].isin(top), df["country"], "Other")

    rhs = "biden"
    if has_type:
        rhs += " + C(doc_type)"
    if has_country:
        rhs += " + C(country_c)"
    if trend:
        rhs += " + year_c + covid"

    rows = []
    for t in range(shares.shape[1]):
        df["y"] = shares[:, t]
        if df["y"].std() == 0:
            continue
        fit = smf.ols(f"y ~ {rhs}", data=df).fit(cov_type="HC3")
        lo, hi = fit.conf_int().loc["biden"]
        rows.append({"topic": t, "biden_coef": fit.params["biden"], "ci_low": lo, "ci_high": hi,
                     "p_value": fit.pvalues["biden"]})
    out = pd.DataFrame(rows)
    out["q_value"] = bh_fdr(out["p_value"].to_numpy())
    out["controls"] = rhs.replace("biden + ", "")
    return out


def lag_robustness(shares: np.ndarray, docs: pd.DataFrame, days: int = cfg.LAG_EXCLUSION_DAYS) -> tuple[pd.DataFrame, int]:
    """
    Recompute the difference after dropping documents dated within `days` of
    each inauguration. Early-term documents were mostly designed under the
    previous administration, so a real shift should get larger, not vanish.
    """
    d = docs["date"]
    early = ((d >= pd.Timestamp(cfg.TRUMP_START)) & (d < pd.Timestamp(cfg.TRUMP_START + timedelta(days=days)))) | \
            ((d >= pd.Timestamp(cfg.BIDEN_START)) & (d < pd.Timestamp(cfg.BIDEN_START + timedelta(days=days))))
    keep = ~early.to_numpy()
    is_b = (docs["administration"] == "Biden").to_numpy()
    obs, p = permutation_test(shares[keep], is_b[keep], n_perm=min(cfg.N_PERMUTATIONS, 5000))
    out = pd.DataFrame({"topic": range(shares.shape[1]), "diff_excl_early": obs, "p_excl_early": p})
    out["q_excl_early"] = bh_fdr(p)
    return out, int(keep.sum())


def yearly_prevalence(shares: np.ndarray, docs: pd.DataFrame) -> pd.DataFrame:
    """Doc-weighted mean share per topic per calendar year, with document counts."""
    years = docs["date"].dt.year.to_numpy()
    rows = []
    for y in sorted(np.unique(years[~np.isnan(years)])):
        m = years == y
        for t, v in enumerate(shares[m].mean(axis=0)):
            rows.append({"year": int(y), "topic": t, "mean_share": v, "n_docs": int(m.sum())})
    return pd.DataFrame(rows)


def run_prevalence(shares: np.ndarray, docs: pd.DataFrame, chunk_topics: np.ndarray, chunk_admin: np.ndarray,
                   topic_info: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    is_b = (docs["administration"] == "Biden").to_numpy()

    table = prevalence_table(shares, docs, chunk_topics, chunk_admin)
    obs, p = permutation_test(shares, is_b)
    table["diff_ci_low"], table["diff_ci_high"] = bootstrap_ci(shares, is_b)
    table["p_perm"] = p
    table["q_perm"] = bh_fdr(p)

    if docs["doc_type"].nunique() > 1:
        _, p_s = permutation_test(shares, is_b, strata=pd.factorize(docs["doc_type"])[0])
        table["p_perm_within_doctype"] = p_s
        table["q_perm_within_doctype"] = bh_fdr(p_s)

    lag, n_kept = lag_robustness(shares, docs)
    table = table.merge(lag, on="topic")
    logger.info("Lag robustness kept %d of %d documents", n_kept, len(docs))

    reg = covariate_regressions(shares, docs)
    if reg is not None:
        reg.to_csv(out_dir / "covariate_regressions.csv", index=False)
        table = table.merge(reg[["topic", "biden_coef", "ci_low", "ci_high", "q_value"]]
                            .rename(columns={"q_value": "q_regression"}), on="topic", how="left")
    reg_t = covariate_regressions(shares, docs, trend=True)
    if reg_t is not None:
        reg_t.to_csv(out_dir / "covariate_regressions_trend.csv", index=False)
        table = table.merge(reg_t[["topic", "biden_coef", "q_value"]]
                            .rename(columns={"biden_coef": "biden_coef_trend", "q_value": "q_trend"}),
                            on="topic", how="left")

    cols = ["topic", "keywords", "gpt_label", "n_docs", "top_doc_share", "single_doc_dominated", "npmi"]
    table = topic_info[[c for c in cols if c in topic_info.columns]].merge(table, on="topic")
    table = table.sort_values("diff_biden_minus_trump", key=np.abs, ascending=False)
    table.to_csv(out_dir / "administration_comparison.csv", index=False)

    yearly_prevalence(shares, docs).to_csv(out_dir / "yearly_prevalence.csv", index=False)
    return table

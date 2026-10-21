"""
Material for reading, not for statistics.

* reading_samples.csv: for each topic, chunks drawn from as many distinct
  documents as possible. Read these before interpreting any topic or
  writing a sentence about why it moved.
* mentions_by_year.csv: share of documents mentioning a term family
  (COVID, AMR, family planning, Mexico City Policy, multilateral partners...).
  This is a direct, label-free check on claims like "multilateral engagement
  fell under Trump" that no topic actually measures.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg


def reading_samples(chunks: pd.DataFrame, topics: np.ndarray, docs: pd.DataFrame, topic_info: pd.DataFrame,
                    out_dir: Path, n: int = cfg.READING_SAMPLE_SIZE) -> None:
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    meta = docs.set_index("doc_id")[["filename", "date", "administration"]]
    df = chunks[["chunk_id", "doc_id", "page", "text"]].copy()
    df["topic"] = topics
    rows = []
    for t, sub in df.groupby("topic"):
        # one chunk per document first, then fill up if the topic has few documents
        per_doc = sub.groupby("doc_id").sample(1, random_state=int(rng.integers(1 << 31)))
        pick = per_doc.sample(min(n, len(per_doc)), random_state=cfg.RANDOM_SEED)
        if len(pick) < n:
            rest = sub.drop(pick.index)
            pick = pd.concat([pick, rest.sample(min(n - len(pick), len(rest)), random_state=cfg.RANDOM_SEED)])
        rows.append(pick)
    out = pd.concat(rows).join(meta, on="doc_id")
    info = topic_info.set_index("topic")
    out["keywords"] = out["topic"].map(info["keywords"])
    if "gpt_label" in info:
        out["gpt_label"] = out["topic"].map(info["gpt_label"])
    out = out.sort_values(["topic", "administration", "date"])
    cols = ["topic", "gpt_label", "keywords", "administration", "date", "filename", "page", "text"]
    out[[c for c in cols if c in out.columns]].to_csv(out_dir / "reading_samples.csv", index=False)


def mentions_by_year(chunks: pd.DataFrame, docs: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    text = chunks.groupby("doc_id")["text"].apply(lambda s: " ".join(s).lower())
    d = docs.set_index("doc_id").join(text.rename("text"))
    d["year"] = d["date"].dt.year
    rows = []
    for name, pattern in cfg.MENTION_TERMS.items():
        rx = re.compile(pattern)
        d[name] = d["text"].fillna("").map(lambda t: bool(rx.search(t)))
    by_year = d.groupby("year")[list(cfg.MENTION_TERMS)].mean()
    by_year["n_docs"] = d.groupby("year").size()
    by_admin = d.groupby("administration")[list(cfg.MENTION_TERMS)].mean()
    by_admin["n_docs"] = d.groupby("administration").size()
    by_year.to_csv(out_dir / "mentions_by_year.csv")
    by_admin.to_csv(out_dir / "mentions_by_administration.csv")
    return by_admin


def mention_indicators(chunks: pd.DataFrame, docs: pd.DataFrame) -> pd.DataFrame:
    """One row per document: True/False for each term family in cfg.MENTION_TERMS."""
    text = chunks.groupby("doc_id")["text"].apply(lambda s: " ".join(s).lower())
    t = docs["doc_id"].map(text).fillna("")
    return pd.DataFrame({name: t.map(lambda s, rx=re.compile(p): bool(rx.search(s)))
                         for name, p in cfg.MENTION_TERMS.items()}, index=docs.index)


def mention_tests(chunks: pd.DataFrame, docs: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """
    For each term family: share of Trump- and Biden-era documents that mention it,
    a document-level permutation test of the difference (BH-FDR across terms), and
    the same trend-adjusted linear probability model used for topics (document
    type, country, linear time trend, COVID period; HC3 SEs). Label-free, so these
    don't depend on any topic-model choice.
    """
    from src.analysis.prevalence import bh_fdr, covariate_regressions, permutation_test

    m = mention_indicators(chunks, docs).astype(float)
    is_b = (docs["administration"] == "Biden").to_numpy()
    obs, p = permutation_test(m.to_numpy(), is_b)
    out = pd.DataFrame({
        "term": m.columns,
        "trump_share": m[~is_b].mean().to_numpy(),
        "biden_share": m[is_b].mean().to_numpy(),
        "diff": obs,
        "n_docs_trump": int((~is_b).sum()),
        "n_docs_biden": int(is_b.sum()),
        "p_perm": p,
        "q_perm": bh_fdr(p),
    })
    reg = covariate_regressions(m.to_numpy(), docs, trend=True)
    if reg is not None:
        reg = reg.rename(columns={"biden_coef": "jump_at_inauguration", "q_value": "q_trend"})
        reg["term"] = m.columns[reg["topic"].to_numpy()]
        out = out.merge(reg[["term", "jump_at_inauguration", "ci_low", "ci_high", "q_trend"]], on="term", how="left")
    out.to_csv(out_dir / "mention_tests.csv", index=False)
    return out

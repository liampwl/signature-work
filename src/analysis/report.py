"""Figures and a plain-markdown summary of a run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg


def _short(row: pd.Series) -> str:
    label = row.get("gpt_label") or ""
    if not isinstance(label, str) or not label:
        label = ", ".join(str(row["keywords"]).split(", ")[:4])
    return f"{int(row['topic'])}: {label}"[:60]


def plot_differences(table: pd.DataFrame, out_dir: Path, n: int = 20) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = table.head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(top) + 1.5))
    y = np.arange(len(top))
    sig = top["q_perm"] < cfg.FDR_ALPHA
    colors = np.where(top["diff_biden_minus_trump"] > 0, "#2b6cb0", "#c53030")
    ax.hlines(y, top["diff_ci_low"] * 100, top["diff_ci_high"] * 100, color="#a0aec0", lw=2)
    ax.scatter(top["diff_biden_minus_trump"] * 100, y, c=colors, s=np.where(sig, 50, 18), zorder=3)
    ax.axvline(0, color="#4a5568", lw=0.8)
    labels = [(_short(r) + ("  [1 doc]" if r.get("single_doc_dominated") else "")) for _, r in top.iterrows()]
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlabel("Biden minus Trump, mean document share (percentage points)\n"
                  "large dots: FDR q < 0.05 (document-level permutation); bars: 95% bootstrap CI")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "administration_differences.png", dpi=200)
    plt.close(fig)


def plot_yearly(yearly: pd.DataFrame, table: pd.DataFrame, out_dir: Path, n: int = 8) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    topics = table.head(n)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for _, r in topics.iterrows():
        sub = yearly[yearly["topic"] == r["topic"]]
        ax.plot(sub["year"], sub["mean_share"] * 100, marker="o", label=_short(r))
    ax.axvline(2021 + 19 / 365, color="#4a5568", ls="--", lw=0.8)
    years = sorted(yearly["year"].unique())
    counts = yearly.drop_duplicates("year").set_index("year")["n_docs"]
    ax.set_xticks(years, [f"{y}\n(n={counts[y]})" for y in years], fontsize=8)
    ax.set_ylabel("Mean document share (%)")
    ax.set_title("Largest administration differences by year (2017 starts Jan 20; last year is partial)")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "yearly_prevalence.png", dpi=200)
    plt.close(fig)


def write_summary(out_dir: Path, docs: pd.DataFrame, chunks: pd.DataFrame, topic_info: pd.DataFrame,
                  table: pd.DataFrame, separability: pd.DataFrame | None, mentions: pd.DataFrame | None,
                  k: int, seeded: bool) -> None:
    n_sig = int((table["q_perm"] < cfg.FDR_ALPHA).sum())
    n_single = int(topic_info["single_doc_dominated"].sum())
    lines = [
        "# Run summary",
        "",
        f"- Documents: {len(docs)} (Trump {int((docs.administration == 'Trump').sum())}, "
        f"Biden {int((docs.administration == 'Biden').sum())})",
        f"- Chunks: {len(chunks)} (median {chunks.n_words.median():.0f} words)",
        f"- K = {k}, guided seeds = {seeded}, seed = {cfg.FINAL_SEED}",
        f"- Largest topic share of chunks: {topic_info.share_of_chunks.max():.1%}",
        f"- Topics where one document supplies >{cfg.SINGLE_DOC_DOMINANCE:.0%} of chunks: {n_single}",
        f"- Mean NPMI (top-{cfg.N_TOP_WORDS} c-TF-IDF unigrams, full corpus): {topic_info.npmi.mean():.3f}",
        f"- Topics with document-level FDR q < {cfg.FDR_ALPHA}: {n_sig} of {len(table)}",
    ]
    for col, name in [("q_perm_within_doctype", "within document type"), ("q_regression", "regression with controls"),
                      ("q_excl_early", f"excluding first {cfg.LAG_EXCLUSION_DAYS} days of each term")]:
        if col in table:
            lines.append(f"  - still q < {cfg.FDR_ALPHA} {name}: {int((table[col] < cfg.FDR_ALPHA).sum())}")
    lines += ["", "## Largest differences (Biden minus Trump, mean document share)", ""]
    show = table.head(20).copy()
    show["diff_pp"] = (show["diff_biden_minus_trump"] * 100).round(2)
    show["ci_pp"] = [f"[{lo*100:.2f}, {hi*100:.2f}]" for lo, hi in zip(show.diff_ci_low, show.diff_ci_high)]
    show["topic_desc"] = [_short(r) for _, r in show.iterrows()]
    cols = ["topic_desc", "diff_pp", "ci_pp", "q_perm", "n_docs", "top_doc_share"]
    for extra in ["q_perm_within_doctype", "q_regression", "q_excl_early"]:
        if extra in show:
            cols.append(extra)
    lines.append(show[cols].round(4).to_markdown(index=False))
    if separability is not None:
        lines += ["", "## Separability (can a classifier tell the periods apart?)", "",
                  separability.round(3).to_markdown(index=False)]
    if mentions is not None:
        lines += ["", "## Share of documents mentioning each term family", "",
                  mentions.round(3).to_markdown()]
    lines += ["", "Read reading_samples.csv before interpreting any topic."]
    (out_dir / "results_summary.md").write_text("\n".join(lines) + "\n")

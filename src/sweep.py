"""
Model-selection sweep over K and random seeds.

Run this and pick K from sweep_summary.csv *before* running the
administration analysis, and write down why. The criteria here are
properties of the topic model only (stability, coherence, diversity,
single-document topics); none of them look at Trump vs. Biden.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

import config as cfg
from src.modeling import TermMatrices, cluster, concentration, get_reduced, topic_diversity
from src.utils import get_logger

logger = get_logger(__name__)


def run_sweep(data_dir: Path, out_dir: Path, chunks: pd.DataFrame, embeddings: np.ndarray,
              seeded: bool, k_grid=None, seeds=None, ari_sample: int = 50_000) -> pd.DataFrame:
    k_grid = k_grid or cfg.SWEEP_K_GRID
    seeds = seeds or cfg.SWEEP_SEEDS
    out_dir.mkdir(parents=True, exist_ok=True)

    terms = TermMatrices(chunks["text"].astype(str).tolist())
    doc_ids = chunks["doc_id"].to_numpy()

    embedding_model = None
    if seeded:
        from src.embedding import load_embedding_model

        embedding_model = load_embedding_model()

    rng = np.random.default_rng(cfg.RANDOM_SEED)
    sample = rng.choice(len(chunks), size=min(ari_sample, len(chunks)), replace=False)

    rows, label_store = [], {}
    for seed in seeds:
        reduced = get_reduced(data_dir, embeddings, seed, seeded, embedding_model)
        for k in k_grid:
            labels = cluster(reduced, k, seed)
            label_store[(k, seed)] = labels[sample]
            scores = terms.ctfidf(labels, k)
            uni = terms.top_words(scores, unigram_only=True)
            npmi = np.array([terms.npmi(w) for w in uni])
            conc = concentration(labels, doc_ids, k)
            sizes = np.bincount(labels, minlength=k) / len(labels)
            rows.append({
                "k": k, "seed": seed,
                "npmi_mean": np.nanmean(npmi),
                "npmi_median": np.nanmedian(npmi),
                "diversity": topic_diversity(terms.top_words(scores)),
                "largest_topic_share": sizes.max(),
                "smallest_topic_share": sizes.min(),
                "n_single_doc_topics": int((conc["top_doc_share"] > cfg.SINGLE_DOC_DOMINANCE).sum()),
                "median_docs_per_topic": float(conc["n_docs"].median()),
            })
            logger.info("k=%d seed=%d  npmi=%.3f  div=%.3f  single-doc=%d",
                        k, seed, rows[-1]["npmi_mean"], rows[-1]["diversity"], rows[-1]["n_single_doc_topics"])

    per_run = pd.DataFrame(rows)
    per_run.to_csv(out_dir / "sweep_runs.csv", index=False)

    # Seed stability: mean pairwise ARI between seeds at the same K
    ari_rows = []
    for k in k_grid:
        aris = [adjusted_rand_score(label_store[(k, a)], label_store[(k, b)]) for a, b in combinations(seeds, 2)]
        ari_rows.append({"k": k, "seed_ari_mean": np.mean(aris) if aris else np.nan,
                         "seed_ari_min": np.min(aris) if aris else np.nan})
    ari = pd.DataFrame(ari_rows)

    summary = per_run.groupby("k").agg(
        npmi_mean=("npmi_mean", "mean"),
        npmi_sd=("npmi_mean", "std"),
        diversity=("diversity", "mean"),
        largest_topic_share=("largest_topic_share", "mean"),
        n_single_doc_topics=("n_single_doc_topics", "mean"),
        median_docs_per_topic=("median_docs_per_topic", "mean"),
    ).reset_index().merge(ari, on="k")
    summary.to_csv(out_dir / "sweep_summary.csv", index=False)
    _plot(summary, out_dir)
    logger.info("Sweep summary:\n%s", summary.round(3).to_string(index=False))
    return summary


def _plot(summary: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        ("seed_ari_mean", "Seed stability (ARI)"),
        ("npmi_mean", "NPMI coherence"),
        ("diversity", "Topic diversity"),
        ("n_single_doc_topics", "Single-document topics"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    for ax, (col, title) in zip(axes, metrics):
        ax.plot(summary["k"], summary[col], marker="o", color="#2b6cb0")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("K")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "sweep_summary.png", dpi=200)
    plt.close(fig)

#!/usr/bin/env python3
"""
Stage 5: administration comparison, robustness checks, separability, reading samples.

    python scripts/05_run_analysis.py --data-dir data --model-dir outputs/final_k40_seeded
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import _common  # noqa: F401
import config as cfg
from src.analysis.inspection import mentions_by_year, reading_samples
from src.analysis.prevalence import run_prevalence
from src.analysis.report import plot_differences, plot_yearly, write_summary
from src.analysis.separability import doc_mean_embeddings, separability
from src.embedding import load_embeddings
from src.utils import doc_topic_shares, get_logger, load_corpus, write_run_info

logger = get_logger(__name__)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=cfg.DATA_DIR)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--skip-separability", action="store_true")
    args = ap.parse_args()
    out = args.model_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    chunks, docs = load_corpus(args.data_dir)
    assign = pd.read_csv(args.model_dir / "topics.csv")
    info = pd.read_csv(args.model_dir / "topic_info.csv")
    info["gpt_label"] = info["gpt_label"].fillna("")
    assert (assign["chunk_id"].to_numpy() == chunks["chunk_id"].to_numpy()).all(), "topics.csv out of sync"
    k = len(info)

    # Analysis population: documents dated inside the two administrations
    docs = docs[docs["administration"].isin(["Trump", "Biden"])].reset_index(drop=True)
    in_scope = chunks["doc_id"].isin(docs["doc_id"]).to_numpy()
    chunks_s = chunks[in_scope].reset_index(drop=True)
    topics_s = assign["topic"].to_numpy()[in_scope]
    admin_of = docs.set_index("doc_id")["administration"]
    chunk_admin = chunks_s["doc_id"].map(admin_of).to_numpy()

    shares = doc_topic_shares(chunks_s, topics_s, docs["doc_id"].to_numpy(), k)
    np.save(out / "doc_topic_shares.npy", shares)

    table = run_prevalence(shares, docs, topics_s, chunk_admin, info, out)
    yearly = pd.read_csv(out / "yearly_prevalence.csv")
    plot_differences(table, out)
    plot_yearly(yearly, table, out)

    sep = None
    if not args.skip_separability:
        from sklearn.preprocessing import StandardScaler

        emb = load_embeddings(args.data_dir, len(chunks))[in_scope]
        doc_emb = doc_mean_embeddings(emb, chunks_s["doc_id"].to_numpy(), docs["doc_id"].to_numpy())
        is_b = (docs["administration"] == "Biden").to_numpy()
        sep = separability(
            {"doc_mean_embedding": doc_emb, "topic_shares": StandardScaler().fit_transform(shares)},
            is_b, out,
        )

    reading_samples(chunks_s, topics_s, docs, info, out)
    mentions = mentions_by_year(chunks_s, docs, out)

    seeded = "unseeded" not in args.model_dir.name
    write_summary(out, docs, chunks_s, info, table, sep, mentions, k, seeded)
    write_run_info(out, {"stage": "analysis", "model_dir": str(args.model_dir)})
    logger.info("Done. See %s", out / "results_summary.md")


if __name__ == "__main__":
    main()

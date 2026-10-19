#!/usr/bin/env python3
"""
Stage 3 — Run all post-training analyses on a saved BERTopic model.

Usage:
    python scripts/03_run_analysis.py \\
        --csv ./data/processed_paragraphs.csv \\
        --model-dir ./outputs/run_001/topic_model \\
        --output-dir ./outputs/run_001_analysis
"""

import argparse
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from bertopic import BERTopic

import config as cfg
from src.analysis.administration import (
    analyze_administration_differences,
    run_null_model_comparison,
)
from src.analysis.coherence import calculate_topic_coherence
from src.analysis.events import detect_focusing_events, visualize_punctuated_equilibrium
from src.analysis.reporting import create_methodological_summary
from src.analysis.similarity import analyze_document_similarity
from src.analysis.temporal import (
    analyze_policy_attention_shifts,
    analyze_temporal_granularity,
    analyze_topic_evolution,
)
from src.preprocessing import load_and_clean_data
from src.utils import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run analyses on a saved BERTopic model.")
    parser.add_argument("--csv", default=cfg.DEFAULT_CSV_PATH, help="Processed paragraphs CSV")
    parser.add_argument(
        "--model-dir",
        default=os.path.join(cfg.DEFAULT_OUTPUT_DIR, "topic_model"),
        help="Path to the saved BERTopic model directory",
    )
    parser.add_argument("--output-dir", default=cfg.DEFAULT_OUTPUT_DIR, help="Directory for analysis outputs")
    parser.add_argument(
        "--permutations", type=int, default=cfg.NULL_MODEL_PERMUTATIONS,
        help="Number of permutations for null model comparison",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load data & model ----
    documents, timestamps, administrations = load_and_clean_data(args.csv, args.output_dir)

    logger.info("Loading saved BERTopic model from %s ...", args.model_dir)
    topic_model = BERTopic.load(args.model_dir)
    topic_info = topic_model.get_topic_info()
    logger.info("Model loaded — %d topics", len(topic_info))

    # Collectors for the summary report
    significance_results = None
    coherence_results = None
    similarity_data = None
    policy_shifts = None
    focusing_events = None
    volatility = None

    # ---- 1. Topic coherence ----
    try:
        coherence_results = calculate_topic_coherence(topic_model, documents, args.output_dir)
    except Exception as exc:
        logger.warning("Topic coherence failed: %s", exc)

    # ---- 2. Topics over time (absolute + normalized) ----
    topics_over_time = None
    try:
        topics_over_time = topic_model.topics_over_time(
            documents, timestamps, nr_bins=cfg.TOPICS_OVER_TIME_BINS,
        )
        topics_over_time.to_csv(os.path.join(args.output_dir, "topics_over_time_absolute.csv"), index=False)

        # Normalize by yearly doc count
        year_counts: dict[int, int] = {}
        for y in timestamps:
            year_counts[y] = year_counts.get(y, 0) + 1

        normalized = topics_over_time.copy()
        for i, row in normalized.iterrows():
            ts = row["Timestamp"]
            year = int(ts) if isinstance(ts, (float, np.float64, int, np.int64)) else int(str(ts).split("-")[0])
            normalized.at[i, "Frequency"] = row["Frequency"] / year_counts.get(year, 1)

        normalized.to_csv(os.path.join(args.output_dir, "topics_over_time_normalized.csv"), index=False)

        topic_model.visualize_topics_over_time(
            normalized, top_n_topics=10, width=1200, height=800,
        ).write_html(os.path.join(args.output_dir, "topics_over_time_normalized.html"))
    except Exception as exc:
        logger.warning("Topics-over-time failed: %s", exc)

    # ---- 3. Standard visualizations ----
    try:
        topic_model.visualize_topics(width=1200, height=800).write_html(
            os.path.join(args.output_dir, "topic_network.html"),
        )
        topic_model.visualize_hierarchy(width=1200, height=800).write_html(
            os.path.join(args.output_dir, "topic_hierarchy.html"),
        )
        topic_model.visualize_barchart(
            top_n_topics=15, n_words=10, width=1200, height=800,
        ).write_html(os.path.join(args.output_dir, "topic_barchart.html"))
    except Exception as exc:
        logger.warning("Visualization generation failed: %s", exc)

    # ---- 4. Administration differences ----
    try:
        analyze_administration_differences(topic_model, documents, administrations, args.output_dir)
    except Exception as exc:
        logger.warning("Administration differences failed: %s", exc)

    # ---- 5. Null model comparison ----
    try:
        significance_results = run_null_model_comparison(
            topic_model, documents, administrations, args.output_dir,
            n_permutations=args.permutations,
        )
    except Exception as exc:
        logger.warning("Null model comparison failed: %s", exc)

    # ---- 6. Document similarity ----
    try:
        analyze_document_similarity(topic_model, documents, administrations, args.output_dir)
        similarity_data = pd.read_csv(os.path.join(args.output_dir, "administration_similarity.csv"))
    except Exception as exc:
        logger.warning("Document similarity failed: %s", exc)

    # ---- 7. Temporal granularity ----
    try:
        analyze_temporal_granularity(topic_model, documents, timestamps, args.output_dir)
    except Exception as exc:
        logger.warning("Temporal granularity failed: %s", exc)

    # ---- 8. Topic evolution ----
    if topics_over_time is not None:
        try:
            analyze_topic_evolution(topic_model, topics_over_time, args.output_dir)
        except Exception as exc:
            logger.warning("Topic evolution failed: %s", exc)

    # ---- 9. Policy attention shifts ----
    try:
        policy_shifts = analyze_policy_attention_shifts(
            topic_model, documents, timestamps, administrations, args.output_dir,
        )
    except Exception as exc:
        logger.warning("Policy attention shifts failed: %s", exc)

    # ---- 10. Focusing events ----
    try:
        focusing_events = detect_focusing_events(topic_model, documents, timestamps, args.output_dir)
    except Exception as exc:
        logger.warning("Focusing events failed: %s", exc)

    # ---- 11. Punctuated equilibrium ----
    if topics_over_time is not None:
        try:
            volatility = visualize_punctuated_equilibrium(topic_model, topics_over_time, args.output_dir)
        except Exception as exc:
            logger.warning("Punctuated equilibrium failed: %s", exc)

    # ---- 12. Summary report ----
    create_methodological_summary(
        args.output_dir,
        significance_results=significance_results,
        coherence_results=coherence_results,
        similarity_data=similarity_data,
        policy_shifts=policy_shifts,
        focusing_events=focusing_events,
        volatility=volatility,
    )

    logger.info("All analyses complete — results in %s", args.output_dir)


if __name__ == "__main__":
    main()

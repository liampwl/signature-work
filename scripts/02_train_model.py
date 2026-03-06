#!/usr/bin/env python3
"""
Stage 2 — Train a guided BERTopic model and save it to disk.

Usage:
    python scripts/02_train_model.py \\
        --csv ./data/processed_paragraphs.csv \\
        --output-dir ./outputs/run_001
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from src.modeling import train_and_save
from src.preprocessing import load_and_clean_data
from src.utils import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train a guided BERTopic model.")
    parser.add_argument("--csv", default=cfg.DEFAULT_CSV_PATH, help="Processed paragraphs CSV")
    parser.add_argument("--output-dir", default=cfg.DEFAULT_OUTPUT_DIR, help="Directory for outputs & saved model")
    args = parser.parse_args()

    documents, timestamps, administrations = load_and_clean_data(args.csv, args.output_dir)

    topic_model, topics, probs = train_and_save(
        documents, timestamps, administrations, args.output_dir,
    )

    logger.info("Training complete — %d topics discovered.", len(set(topics)))


if __name__ == "__main__":
    main()

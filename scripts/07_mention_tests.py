#!/usr/bin/env python3
"""
Term-mention tests on their own (no topic model needed): share of documents
mentioning each term family in config.MENTION_TERMS, by administration and year,
with document-level permutation tests and the trend-adjusted model.

    python scripts/07_mention_tests.py --data-dir data --out-dir outputs/mentions
"""
import argparse
from pathlib import Path

import _common  # noqa: F401
import config as cfg
from src.analysis.inspection import mention_tests, mentions_by_year
from src.utils import load_corpus


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=cfg.DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=cfg.OUTPUT_DIR / "mentions")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    chunks, docs = load_corpus(args.data_dir)
    docs = docs[docs["administration"].isin(["Trump", "Biden"])].reset_index(drop=True)
    chunks = chunks[chunks["doc_id"].isin(docs["doc_id"])]
    mentions_by_year(chunks, docs, args.out_dir)
    print(mention_tests(chunks, docs, args.out_dir).round(4).to_string(index=False))


if __name__ == "__main__":
    main()

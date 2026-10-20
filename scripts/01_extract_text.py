#!/usr/bin/env python3
"""
Stage 1: PDFs -> chunks.csv, documents.csv, manifest.csv, filter_counts.csv

    python scripts/01_extract_text.py --pdf-dir /path/to/pdfs --data-dir data [--metadata data/dec_metadata.csv]
"""
import argparse
from pathlib import Path

import _common  # noqa: F401
import config as cfg
from src.preprocessing import attach_metadata, extract_corpus


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf-dir", type=Path, default=cfg.PDF_DIR)
    ap.add_argument("--data-dir", type=Path, default=cfg.DATA_DIR)
    ap.add_argument("--metadata", default=cfg.METADATA_CSV, help="DEC export CSV (optional)")
    ap.add_argument("--metadata-only", action="store_true",
                    help="re-attach --metadata to an existing documents.csv without re-extracting")
    args = ap.parse_args()
    if args.metadata_only:
        attach_metadata(args.data_dir, args.metadata)
    else:
        extract_corpus(args.pdf_dir, args.data_dir, args.metadata)


if __name__ == "__main__":
    main()

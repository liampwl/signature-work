#!/usr/bin/env python3
"""
Stage 1 — Extract text from USAID policy PDFs into a structured CSV.

Usage:
    python scripts/01_extract_text.py --pdf-dir ./data/pdfs --output ./data/processed_paragraphs.csv
"""

import argparse
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from src.preprocessing import extract_text_from_pdfs


def main():
    parser = argparse.ArgumentParser(description="Extract paragraphs from PDFs via OCR.")
    parser.add_argument("--pdf-dir", default=cfg.DEFAULT_PDF_DIR, help="Directory containing source PDFs")
    parser.add_argument("--output", default=cfg.DEFAULT_CSV_PATH, help="Path for the output CSV")
    parser.add_argument("--batch-size", type=int, default=50, help="PDFs per processing batch")
    args = parser.parse_args()

    extract_text_from_pdfs(args.pdf_dir, args.output, batch_size=args.batch_size)


if __name__ == "__main__":
    main()

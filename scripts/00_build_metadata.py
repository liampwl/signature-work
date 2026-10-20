#!/usr/bin/env python3
"""
Stage 0: rebuild the PDF filename -> DEC metadata mapping.

The thesis download script ("pilot download script.py") saved row i of the
finalized DEC CSV as  document_{Publication Date as YYYY-MM-DD}_{i+1}.pdf.
This reproduces that naming so every PDF gets its DEC document type,
geography, primary subject and DEC Unique_ID.

    python scripts/00_build_metadata.py --dec-csv "SW DEC Database finalized jan9.csv" \
        --pdf-dir pdfs --out data/dec_metadata.csv
"""
import argparse
from pathlib import Path

import pandas as pd

import _common  # noqa: F401


def geography(value) -> str:
    if not isinstance(value, str) or not value.strip():
        return "Unspecified"
    parts = [p.strip() for p in value.split("|") if p.strip()]
    return parts[0] if len(parts) == 1 else "Multi-country/Regional"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dec-csv", type=Path, required=True)
    ap.add_argument("--pdf-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--date-format", default="%m/%d/%y", help="format of 'Publication Date' in the DEC CSV")
    args = ap.parse_args()

    dec = pd.read_csv(args.dec_csv, dtype=str, keep_default_na=False, encoding_errors="replace")
    dates = pd.to_datetime(dec["Publication Date"], format=args.date_format, errors="coerce")
    dec["filename"] = [
        f"document_{d.strftime('%Y-%m-%d')}_{i + 1}.pdf" if pd.notna(d) else ""
        for i, d in enumerate(dates)
    ]
    out = pd.DataFrame({
        "filename": dec["filename"],
        "doc_type": dec.get("Bibliographic_Type", "Unknown"),
        "country": dec.get("Descriptors_Geographic", "").map(geography),
        "geography_raw": dec.get("Descriptors_Geographic", ""),
        "primary_subject": dec.get("Primary_Subject", ""),
        "dec_unique_id": dec.get("Unique_ID", ""),
        "title": dec.get("Title", ""),
        "dec_url": dec.get("File", ""),
    })
    out = out[out["filename"] != ""]

    on_disk = {p.name for p in args.pdf_dir.glob("*.pdf")}
    matched = out["filename"].isin(on_disk)
    print(f"DEC rows: {len(dec)} | with parseable dates: {len(out)} | matched to PDFs on disk: {matched.sum()} "
          f"| PDFs without a DEC row: {len(on_disk - set(out['filename']))}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()

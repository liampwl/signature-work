#!/usr/bin/env python3
"""
Stage 0: rebuild the PDF filename -> DEC metadata mapping.

The thesis download script ("pilot download script.py") saved row i of the
finalized DEC CSV as  document_{Publication Date as YYYY-MM-DD}_{i+1}.pdf.
This reproduces that naming so every PDF gets its DEC document type,
geography, primary subject and DEC Unique_ID.

Rows whose Publication Date was only a year were downloaded later under a
different numbering. With --match-by-title, each unmatched PDF is matched to
the year-only DEC row (same year) whose title words best appear on the PDF's
first two pages; matches below --min-title-score are left unmatched.

    python scripts/00_build_metadata.py --dec-csv "SW DEC Database finalized jan9.csv" \
        --pdf-dir pdfs --out data/dec_metadata.csv --match-by-title
"""
import argparse
import re
from pathlib import Path

import pandas as pd

import _common  # noqa: F401


def geography(value) -> str:
    if not isinstance(value, str) or not value.strip():
        return "Unspecified"
    parts = [p.strip() for p in value.split("|") if p.strip()]
    return parts[0] if len(parts) == 1 else "Multi-country/Regional"


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{3,}", text.lower()))


def match_by_title(unmatched: list[str], dec: pd.DataFrame, pdf_dir: Path, min_score: float) -> pd.DataFrame:
    try:
        import pymupdf
    except ImportError:
        import fitz as pymupdf
    year_only = dec[dec["Publication Date"].str.fullmatch(r"\d{4}")]
    rows = []
    for f in unmatched:
        year = f.split("_")[1][:4]
        try:
            with pymupdf.open(pdf_dir / f) as doc:
                text = " ".join(doc[i].get_text() for i in range(min(2, len(doc))))
        except Exception:
            text = ""
        words = _tokens(text)
        best = (0.0, None)
        for idx, r in year_only[year_only["Publication Date"] == year].iterrows():
            title = _tokens(r["Title"])
            if title:
                score = len(title & words) / len(title)
                if score > best[0]:
                    best = (score, idx)
        rows.append({"filename": f, "score": best[0], "dec_index": best[1]})
    m = pd.DataFrame(rows)
    m = m[m["score"] >= min_score]
    # one DEC row per PDF: keep the best-scoring PDF for each row
    m = m.sort_values("score", ascending=False).drop_duplicates("dec_index")
    return m


def build_rows(dec: pd.DataFrame, filenames: pd.Series, how: str) -> pd.DataFrame:
    geo = dec.get("Descriptors_Geographic", pd.Series("", index=dec.index))
    return pd.DataFrame({
        "filename": filenames.values,
        # DEC uses two spellings for the same type
        "doc_type": dec.get("Bibliographic_Type", pd.Series("Unknown", index=dec.index))
        .str.replace("Project Program Overview", "Project/Program Overview", regex=False).values,
        "country": geo.map(geography).values,
        "geography_raw": geo.values,
        "primary_subject": dec.get("Primary_Subject", pd.Series("", index=dec.index)).values,
        "dec_unique_id": dec.get("Unique_ID", pd.Series("", index=dec.index)).values,
        "title": dec.get("Title", pd.Series("", index=dec.index)).values,
        "dec_publication_date": dec["Publication Date"].values,
        "match_method": how,
    })


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dec-csv", type=Path, required=True)
    ap.add_argument("--pdf-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--date-format", default="%m/%d/%y", help="format of 'Publication Date' in the DEC CSV")
    ap.add_argument("--match-by-title", action="store_true")
    ap.add_argument("--min-title-score", type=float, default=0.8)
    args = ap.parse_args()

    dec = pd.read_csv(args.dec_csv, dtype=str, keep_default_na=False, encoding_errors="replace")
    dates = pd.to_datetime(dec["Publication Date"], format=args.date_format, errors="coerce")
    names = pd.Series(
        [f"document_{d.strftime('%Y-%m-%d')}_{i + 1}.pdf" if pd.notna(d) else "" for i, d in enumerate(dates)],
        index=dec.index,
    )
    on_disk = {p.name for p in args.pdf_dir.glob("*.pdf")}
    keep = names.isin(on_disk)
    out = build_rows(dec[keep], names[keep], "row_index")

    unmatched = sorted(on_disk - set(out["filename"]))
    if args.match_by_title and unmatched:
        m = match_by_title(unmatched, dec, args.pdf_dir, args.min_title_score)
        extra = build_rows(dec.loc[m["dec_index"]], m["filename"], "title")
        extra["title_score"] = m["score"].values
        out = pd.concat([out, extra], ignore_index=True)
        print(f"Title matching: {len(m)} of {len(unmatched)} unmatched PDFs matched (score >= {args.min_title_score})")

    print(f"DEC rows: {len(dec)} | PDFs on disk: {len(on_disk)} | PDFs with metadata: {out['filename'].nunique()} "
          f"| without: {len(on_disk - set(out['filename']))}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()

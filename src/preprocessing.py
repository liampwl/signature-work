"""
Stage 1: PDF -> paragraphs -> embedding-sized chunks.

What changed from the thesis (v1) pipeline, and why:

* Paragraphs come from PyMuPDF text *blocks*, not from splitting page text on
  blank lines. PyMuPDF's plain-text output rarely contains blank lines, so v1
  produced roughly one "paragraph" per page.
* Running headers/footers are detected per document and removed instead of
  being embedded on every page.
* Long paragraphs are split at sentence boundaries so that chunks fit the
  embedding model's context window (v1 silently truncated at 128 tokens).
* Filters no longer drop prose that starts with "However," etc., and no longer
  delete everything in parentheses (which removed acronyms like PEPFAR).
* Language detection is seeded, keeps anything not confidently non-English,
  and is applied per paragraph.
* Full publication dates (not just the year) drive the administration label,
  so 1-19 January 2021 is no longer counted as Biden.
* A manifest with MD5 hashes records every PDF and what happened to it, so the
  corpus can be archived and reconstructed now that the DEC is offline.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import config as cfg
from src.utils import get_logger

logger = get_logger(__name__)

try:
    import pymupdf as fitz  # PyMuPDF >= 1.24
except ImportError:  # pragma: no cover
    import fitz

from langdetect import DetectorFactory, LangDetectException, detect_langs

DetectorFactory.seed = cfg.LANGDETECT_SEED


# ---------------------------------------------------------------------------
# Dates and administrations
# ---------------------------------------------------------------------------

_FILENAME_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def date_from_filename(filename: str) -> date | None:
    """Parse the first YYYY-MM-DD in a filename (v1 files: document_YYYY-MM-DD_XXX.pdf)."""
    m = _FILENAME_DATE.search(filename)
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def administration_for(d: date | None) -> str:
    if d is None:
        return "Unknown"
    # 2021-01-01 is what the thesis date-cleaning assigned to year-only and
    # month-only January 2021 documents, so it can't be placed on either side
    # of the 20 January inauguration. Excluded from comparisons.
    if d == date(2021, 1, 1):
        return "Ambiguous"
    if cfg.TRUMP_START <= d < cfg.BIDEN_START:
        return "Trump"
    if cfg.BIDEN_START <= d < cfg.BIDEN_END:
        return "Biden"
    return "Other"


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\S+@\S+\.\w+")
_DOT_LEADER = re.compile(r"(?:\s?\.){4,}")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_BULLETS = re.compile(r"^[•●▪■–—\-\*o]\s+", re.M)


_BAD_CATEGORIES = {"Cc", "Co", "Cs", "Cn"}  # control, private-use (symbol-font glyphs), surrogate, unassigned


def strip_bad_chars(text: str) -> str:
    """Replace control and private-use characters (common in PDF symbol fonts) with spaces."""
    return "".join(
        " " if (unicodedata.category(ch) in _BAD_CATEGORIES and ch != "\n") else ch for ch in text
    )


def normalise_block(text: str) -> str:
    """Unicode-normalise, repair line-break hyphenation, and flatten a block to one line."""
    text = strip_bad_chars(unicodedata.normalize("NFKC", text))
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _BULLETS.sub("", text)
    text = text.replace("\n", " ")
    text = _URL.sub(" ", text)
    text = _EMAIL.sub(" ", text)
    text = _DOT_LEADER.sub(" ", text)
    return " ".join(text.split())


def _header_key(text: str) -> str:
    """Key used to spot repeated headers/footers: lowercase, digits collapsed."""
    return re.sub(r"\d+", "#", text.lower()).strip()


# ---------------------------------------------------------------------------
# Paragraph filters
# ---------------------------------------------------------------------------

_TOC_LINE = re.compile(r"(?:\s?\.){4,}\s*\d+\s*$")
_REF_SIGNALS = re.compile(
    r"\bet al\.|\bdoi\b|\bretrieved from\b|\bpp\.\s*\d|\bvol\.\s*\d|\bjournal of\b|\(\s*(?:19|20)\d{2}[a-z]?\s*\)",
    re.I,
)
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_NON_LATIN = re.compile(r"[^\x00-ɏḀ-ỿ -⁯₠-⃏℀-⅏]")
_SECTION_STOP = re.compile(
    r"^(references|bibliography|works cited|endnotes|acronyms( and abbreviations)?|abbreviations|list of acronyms)$",
    re.I,
)


@dataclass
class FilterCounts:
    counts: Counter = field(default_factory=Counter)

    def hit(self, reason: str) -> bool:
        self.counts[reason] += 1
        return False


def keep_paragraph(text: str, fc: FilterCounts) -> bool:
    words = text.split()
    n_words = len(words)
    if n_words < cfg.MIN_PARAGRAPH_WORDS:
        return fc.hit("short")

    letters = [c for c in text if c.isalpha()]
    if not letters:
        return fc.hit("no_letters")
    if sum(1 for c in letters if _NON_LATIN.match(c)) / len(letters) > 0.2:
        return fc.hit("non_latin_script")

    alnum = sum(1 for c in text if c.isalnum())
    if alnum and sum(1 for c in text if c.isdigit()) / alnum > 0.3:
        return fc.hit("numeric_table")

    if sum(1 for c in letters if c.isupper()) / len(letters) > 0.8:
        return fc.hit("all_caps")

    # Reference-list entries: short, carry a year, and carry a citation signal.
    if n_words < 80 and _YEAR.search(text) and _REF_SIGNALS.search(text):
        return fc.hit("reference_entry")

    try:
        langs = detect_langs(text)
        top = langs[0]
        if top.lang != "en" and top.prob >= cfg.NON_ENGLISH_MIN_PROB:
            return fc.hit(f"language_{top.lang}")
    except LangDetectException:
        pass

    return True


# ---------------------------------------------------------------------------
# Sentence-aware chunking
# ---------------------------------------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9\"“(])")


def chunk_paragraph(text: str, max_words: int = cfg.MAX_CHUNK_WORDS) -> list[str]:
    """Split a paragraph into chunks of <= max_words, breaking at sentence ends where possible."""
    words = text.split()
    if len(words) <= max_words:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    for sent in _SENT_SPLIT.split(text):
        sw = sent.split()
        # Sentence longer than the limit: hard-split it.
        while len(sw) > max_words:
            if current:
                chunks.append(" ".join(current))
                current = []
            chunks.append(" ".join(sw[:max_words]))
            sw = sw[max_words:]
        if len(current) + len(sw) > max_words and current:
            chunks.append(" ".join(current))
            current = []
        current.extend(sw)
    if current:
        # Fold a tiny tail into the previous chunk rather than emitting a fragment.
        if chunks and len(current) < cfg.MIN_PARAGRAPH_WORDS and len(chunks[-1].split()) + len(current) <= max_words * 1.2:
            chunks[-1] = chunks[-1] + " " + " ".join(current)
        else:
            chunks.append(" ".join(current))
    return chunks


# ---------------------------------------------------------------------------
# Per-PDF extraction
# ---------------------------------------------------------------------------

def _ends_sentence(text: str) -> bool:
    return bool(re.search(r"[.!?\"”')\]]\s*$", text))


def extract_paragraphs(pdf_path: Path, fc: FilterCounts) -> tuple[list[dict], dict]:
    """
    Return (paragraph records, per-document info) for one PDF.

    Paragraph = one text block, merged with the next block when a sentence
    clearly runs across a column or page break.
    """
    info = {"n_pages": 0, "n_blocks": 0, "n_header_footer_blocks": 0, "status": "ok"}
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:  # corrupt / encrypted
        info["status"] = f"unreadable: {type(exc).__name__}"
        return [], info

    with doc:
        info["n_pages"] = len(doc)
        pages: list[list[tuple[float, float, str]]] = []
        for page in doc:
            h = page.rect.height or 1.0
            blocks = []
            for b in page.get_text("blocks", sort=True):
                x0, y0, x1, y1, text, _bno, btype = b[:7]
                if btype != 0 or not text.strip():
                    continue
                blocks.append((y0 / h, y1 / h, text))
            pages.append(blocks)

    info["n_blocks"] = sum(len(p) for p in pages)
    if info["n_blocks"] == 0:
        info["status"] = "no_text_layer"  # likely a scan; needs OCR
        return [], info

    # Running headers/footers: blocks whose digit-normalised text repeats on
    # many pages of this document.
    n_pages = len(pages)
    key_pages = Counter()
    for p in pages:
        for k in {_header_key(normalise_block(t)) for _, _, t in p}:
            key_pages[k] += 1
    repeated = (
        {k for k, n in key_pages.items() if n >= max(3, cfg.HEADER_FOOTER_MIN_PAGE_SHARE * n_pages)}
        if n_pages >= 4
        else set()
    )

    paragraphs: list[dict] = []
    pending: dict | None = None
    in_skip_section = False

    for page_no, blocks in enumerate(pages, start=1):
        for top, bottom, raw in blocks:
            text = normalise_block(raw)
            if not text:
                continue
            key = _header_key(text)
            if key in repeated:
                info["n_header_footer_blocks"] += 1
                continue
            # Section headings that start a reference list / acronym list.
            # Everything after is skipped until a long prose block appears.
            if _SECTION_STOP.match(text.strip(" :.").lower()):
                in_skip_section = True
                fc.hit("section_heading_refs_or_acronyms")
                continue
            # Bare page numbers / short margin text
            if (top < 0.06 or bottom > 0.94) and len(text.split()) <= 12:
                info["n_header_footer_blocks"] += 1
                continue
            # Table-of-contents lines
            if _TOC_LINE.search(raw.strip()):
                fc.hit("table_of_contents")
                continue
            if in_skip_section:
                # A long prose block ends the skip (annex content after references).
                if len(text.split()) >= 40 and not _REF_SIGNALS.search(text):
                    in_skip_section = False
                else:
                    fc.hit("in_refs_or_acronym_section")
                    continue

            # Merge with the previous block if the sentence runs on: the previous
            # block has no closing punctuation and either the next block starts
            # lowercase or the previous block is too long to be a heading.
            if pending and not _ends_sentence(pending["text"]) and (
                text[:1].islower() or len(pending["text"].split()) >= 12
            ):
                pending["text"] += " " + text
                continue
            if pending:
                paragraphs.append(pending)
            pending = {"page": page_no, "text": text}
    if pending:
        paragraphs.append(pending)

    return paragraphs, info


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Optional DEC metadata
# ---------------------------------------------------------------------------

_DOCTYPE_GUESSES = ["doc_type", "Bibliographic_Type", "Bibtype_Name", "bibtype_name", "Document Type", "document_type", "Bibtype", "Type"]
_COUNTRY_GUESSES = ["country", "Country", "country", "Geographic_Area", "Geography", "Countries", "Primary_Country"]
_FILE_GUESSES = ["filename", "file_name", "File", "pdf", "local_filename"]


def _pick(cols: list[str], explicit: str | None, guesses: list[str]) -> str | None:
    if explicit:
        return explicit if explicit in cols else None
    for g in guesses:
        if g in cols:
            return g
    return None


def load_metadata(path: str | None) -> pd.DataFrame | None:
    """Load the DEC export and return columns: filename, doc_type, country (where available)."""
    if not path:
        return None
    meta = pd.read_csv(path, dtype=str, low_memory=False)
    cols = list(meta.columns)
    fcol = _pick(cols, cfg.META_FILENAME_COL, _FILE_GUESSES)
    if fcol is None:
        logger.warning(
            "Metadata CSV has no filename column I recognise (%s). Set META_FILENAME_COL.", cols
        )
        return None
    out = pd.DataFrame({"filename": meta[fcol].astype(str).str.strip()})
    dcol = _pick(cols, cfg.META_DOCTYPE_COL, _DOCTYPE_GUESSES)
    ccol = _pick(cols, cfg.META_COUNTRY_COL, _COUNTRY_GUESSES)
    out["doc_type"] = meta[dcol].fillna("Unknown") if dcol else "Unknown"
    out["country"] = meta[ccol].fillna("Unknown") if ccol else "Unknown"
    logger.info("Metadata: filename=%s doc_type=%s country=%s", fcol, dcol, ccol)
    return out.drop_duplicates("filename")


def _merge_metadata(docs: pd.DataFrame, meta: pd.DataFrame | None) -> pd.DataFrame:
    docs = docs.drop(columns=[c for c in ("doc_type", "country") if c in docs.columns])
    if meta is None:
        docs["doc_type"] = "Unknown"
        docs["country"] = "Unknown"
        return docs
    docs = docs.merge(meta[["filename", "doc_type", "country"]], on="filename", how="left")
    missing = docs["doc_type"].isna().sum()
    if missing:
        logger.warning("%d documents had no metadata match", missing)
    docs[["doc_type", "country"]] = docs[["doc_type", "country"]].fillna("Unknown")
    return docs


def attach_metadata(data_dir: Path, metadata_csv: str) -> None:
    """Re-attach DEC metadata to an existing documents.csv without re-extracting."""
    docs = pd.read_csv(data_dir / "documents.csv")
    docs = _merge_metadata(docs, load_metadata(metadata_csv))
    docs.to_csv(data_dir / "documents.csv", index=False)
    logger.info("Document types:\n%s", docs["doc_type"].value_counts().to_string())


# ---------------------------------------------------------------------------
# Corpus driver
# ---------------------------------------------------------------------------

def extract_corpus(pdf_dir: Path, out_dir: Path, metadata_csv: str | None = None,
                   time_budget: float | None = None) -> None:
    """
    Writes to out_dir:
      chunks.csv          one row per chunk (the unit that gets embedded)
      documents.csv       one row per kept document, with date/admin/metadata
      manifest.csv        every PDF seen: md5, status, counts (archive this)
      filter_counts.csv   paragraphs removed by each rule
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(p for p in Path(pdf_dir).rglob("*") if p.suffix.lower() == ".pdf")
    logger.info("Found %d PDFs under %s", len(pdfs), pdf_dir)

    fc = FilterCounts()
    manifest, chunk_rows, seen_md5 = [], [], {}

    # Per-PDF cache so a long extraction can be stopped and resumed
    # (--time-budget) without redoing finished files.
    cache_dir = out_dir / "extract_cache"
    cache_dir.mkdir(exist_ok=True)
    started = time.monotonic()
    complete = True
    md5_cache_path = out_dir / "md5_cache.json"
    md5_cache = json.loads(md5_cache_path.read_text()) if md5_cache_path.exists() else {}

    for pdf in tqdm(pdfs, desc="Extracting"):
        st = pdf.stat()
        key = f"{pdf.name}|{st.st_size}|{int(st.st_mtime)}"
        md5 = md5_cache.get(key) or _md5(pdf)
        md5_cache[key] = md5
        d = date_from_filename(pdf.name)
        row = {
            "filename": pdf.name,
            "relpath": str(pdf.relative_to(pdf_dir)),
            "md5": md5,
            "date": d.isoformat() if d else "",
            "administration": administration_for(d),
        }
        if md5 in seen_md5:
            row.update(status=f"duplicate_of:{seen_md5[md5]}", n_pages=0, n_paragraphs=0, n_chunks=0)
            manifest.append(row)
            continue
        seen_md5[md5] = pdf.name

        cached = cache_dir / f"{md5}.json"
        if cached.exists():
            payload = json.loads(cached.read_text())
        else:
            if time_budget and time.monotonic() - started > time_budget:
                complete = False
                break
            local = FilterCounts()
            paras, info = extract_paragraphs(pdf, local)
            kept_ = [p for p in paras if keep_paragraph(p["text"], local)]
            payload = {"info": info, "kept": kept_, "counts": dict(local.counts)}
            cached.write_text(json.dumps(payload))
        info, kept = payload["info"], payload["kept"]
        for p in kept:  # also cleans caches written before strip_bad_chars existed
            p["text"] = " ".join(strip_bad_chars(p["text"]).split())
        kept = [p for p in kept if len(p["text"].split()) >= cfg.MIN_PARAGRAPH_WORDS]
        fc.counts.update(payload["counts"])
        doc_id = len(seen_md5) - 1
        n_chunks = 0
        for p_idx, p in enumerate(kept):
            for c_idx, chunk in enumerate(chunk_paragraph(p["text"])):
                if len(chunk.split()) < 5:  # fragments left over from hard splits
                    continue
                chunk_rows.append({
                    "doc_id": doc_id,
                    "filename": pdf.name,
                    "page": p["page"],
                    "para_idx": p_idx,
                    "chunk_idx": c_idx,
                    "n_words": len(chunk.split()),
                    "text": chunk,
                })
                n_chunks += 1

        status = info["status"]
        if status == "ok" and n_chunks == 0:
            status = "no_usable_text"
        row.update(
            doc_id=doc_id, status=status, n_pages=info["n_pages"],
            n_header_footer_blocks=info["n_header_footer_blocks"],
            n_paragraphs=len(kept), n_chunks=n_chunks,
        )
        manifest.append(row)

    md5_cache_path.write_text(json.dumps(md5_cache))
    if not complete:
        logger.info("Time budget reached after %d of %d PDFs; run again to continue.", len(manifest), len(pdfs))
        return

    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(out_dir / "manifest.csv", index=False)

    chunks = pd.DataFrame(chunk_rows)
    docs = manifest_df[(manifest_df["status"] == "ok")].copy()
    docs = docs[["doc_id", "filename", "md5", "date", "administration", "n_pages", "n_paragraphs", "n_chunks"]]
    docs["doc_id"] = docs["doc_id"].astype(int)

    docs = _merge_metadata(docs, load_metadata(metadata_csv))

    # Keep only chunks from usable documents
    chunks = chunks[chunks["doc_id"].isin(docs["doc_id"])].reset_index(drop=True)
    chunks.insert(0, "chunk_id", range(len(chunks)))

    chunks.to_csv(out_dir / "chunks.csv", index=False)
    docs.to_csv(out_dir / "documents.csv", index=False)
    pd.DataFrame(sorted(fc.counts.items()), columns=["rule", "paragraphs_removed"]).to_csv(
        out_dir / "filter_counts.csv", index=False
    )

    logger.info("Manifest status counts:\n%s", manifest_df["status"].value_counts().to_string())
    logger.info("Documents kept by administration:\n%s", docs["administration"].value_counts().to_string())
    logger.info(
        "Chunks: %d | median words %.0f | docs %d",
        len(chunks), chunks["n_words"].median() if len(chunks) else 0, len(docs),
    )

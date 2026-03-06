"""
Text preprocessing: PDF extraction (OCR), text cleaning, and stopword management.

Combines functionality from the original `ocr revision for a new csv.py` and the
shared `load_and_clean_data` / `get_enhanced_stopwords` helpers.
"""

from __future__ import annotations

import csv
import os
import re
from typing import List, Tuple

import fitz  # PyMuPDF
import pandas as pd
from langdetect import LangDetectException, detect
from nltk.corpus import stopwords
from tqdm import tqdm

from src.utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------

def get_enhanced_stopwords() -> list[str]:
    """
    Comprehensive stopword list for USAID policy document analysis.

    Includes standard English stopwords plus domain-specific terms that add
    noise to topic models (publication boilerplate, date references, document
    structure terms, non-English fragments, etc.).
    """
    basic = list(stopwords.words("english"))

    publication = [
        "prepared", "publication", "suggested", "citation", "isbn", "report",
        "washington", "dc", "agreement", "review", "official",
        "views", "positions", "contents", "executive", "summary",
        "background", "overview", "methodology", "appendix", "annex",
    ]

    structure = [
        "page", "section", "chapter", "paragraph", "figure", "table",
        "reference", "bibliography", "footnote", "note", "volume", "https", "www",
    ]

    time_terms = [
        "january", "february", "march", "april", "may", "june", "july",
        "august", "september", "october", "november", "december",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct",
        "nov", "dec", "month", "monthly", "quarter", "quarterly", "annual",
        "annually", "year", "years", "fiscal", "fy",
        "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023",
        "fy17", "fy18",
    ]

    non_english_fragments = [
        "pennsylvania ave", "de", "la", "les", "en", "des", "et", "le",
        "de la", "du", "para", "dan", "pa", "za", "ke", "ali", "tes",
        "primer", "wa", "ts", "ee", "tr", "ea", "ti", "te", "ie", "io",
        "ro", "sb", "fn", "aa", "uo", "ye", "na na na", "na", "na na",
        "tbd", "90 90", "600", "200", "85", "90",
    ]

    admin = [
        "total", "number", "percent", "frequency", "details",
        "questionnaire", "respondents", "procedure", "guidelines",
    ]

    org = [
        "outreach", "reference", "budget", "procurement",
        "cooperative", "agreement", "implemented",
    ]

    combined = basic + publication + structure + time_terms + non_english_fragments + admin + org
    return list(set(combined))


# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------

_CLEANING_PATTERNS = [
    r"ISBN-: ----",
    r"AID-OAA-A--",
    r"\s{2,}",
    r"\.{3,}",
    r"\([^\)]*\)",
    r"\d+\.\d+\.\d+",
    r"^\s*\d+\.",
    r"Figure \d+",
    r"Table \d+",
]


def clean_text(text: str) -> str:
    """Apply regex-based cleaning to a single document string."""
    for pattern in _CLEANING_PATTERNS:
        text = re.sub(pattern, " ", text)
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# CSV data loading (for stages 2 & 3)
# ---------------------------------------------------------------------------

def load_and_clean_data(
    file_path: str,
    output_dir: str,
) -> Tuple[List[str], List[int], List[str]]:
    """
    Load the paragraph CSV, clean text, assign administration labels, and
    save yearly paragraph counts.

    Returns (documents, timestamps, administrations).
    """
    logger.info("Loading data from %s …", file_path)
    df = pd.read_csv(file_path, dtype={"year": int}, low_memory=False)
    logger.info("Loaded %d paragraphs  |  columns: %s", len(df), df.columns.tolist())

    # Clean text
    tqdm.pandas(desc="Cleaning text")
    df["text"] = df["text"].astype(str).progress_apply(clean_text)

    # Administration labels
    df["administration"] = df["year"].apply(
        lambda y: "Biden" if y >= 2021 else ("Trump" if y >= 2017 else "Other")
    )

    # Log stats
    logger.info("Year range: %d – %d", df["year"].min(), df["year"].max())
    for admin, count in df["administration"].value_counts().items():
        logger.info("  %s: %d paragraphs", admin, count)

    # Save yearly counts
    yearly = (
        df["year"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "year", "year": "num_paragraphs"})
        .sort_values("year")
    )
    os.makedirs(output_dir, exist_ok=True)
    yearly_path = os.path.join(output_dir, "yearly_paragraph_counts.csv")
    yearly.to_csv(yearly_path, index=False)
    logger.info("Saved yearly counts → %s", yearly_path)

    return df["text"].tolist(), df["year"].tolist(), df["administration"].tolist()


# ---------------------------------------------------------------------------
# PDF / OCR extraction (stage 1)
# ---------------------------------------------------------------------------

# Languages to skip during paragraph filtering
_UNWANTED_LANGUAGES = {"id", "fr", "es", "ar", "th"}


def _is_valid_paragraph(text: str, min_words: int = 10, counters: dict | None = None) -> bool:
    """Determine whether an extracted paragraph should be kept."""
    c = counters  # shorthand; may be None

    if "." * 80 in text:
        if c: c["table_of_contents"] += 1
        return False
    if "ACRONYMS AND ABBREVIATIONS" in text:
        if c: c["acronyms"] += 1
        return False
    if re.search(r"[\u0E00-\u0E7F\u0600-\u06FF]", text):
        if c: c["non_latin"] += 1
        return False
    if len(text.split()) < min_words:
        if c: c["short_paragraph"] += 1
        return False
    if len(re.findall(r"\d", text)) / max(len(text), 1) > 0.2:
        if c: c["too_many_numbers"] += 1
        return False
    if text.isupper():
        if c: c["all_caps"] += 1
        return False
    if re.match(r"^(\d+\.|[A-Z][a-z]+\s*,|et al\.)", text):
        if c: c["references"] += 1
        return False
    try:
        if detect(text) in _UNWANTED_LANGUAGES:
            if c: c["unwanted_language"] += 1
            return False
    except LangDetectException:
        pass

    return True


def _extract_paragraphs_from_pdf(pdf_path: str, counters: dict | None = None) -> list[str]:
    """Extract and filter paragraphs from a single PDF."""
    pdf_path = str(pdf_path)
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

    paragraphs: list[str] = []
    try:
        with fitz.open(pdf_path) as doc:
            for page_num in range(len(doc)):
                try:
                    text = doc[page_num].get_textpage().extractText()
                    for p in text.split("\n\n"):
                        p = p.strip()
                        if p and _is_valid_paragraph(p, counters=counters):
                            paragraphs.append(p)
                except Exception as e:
                    logger.warning("Skipping page %d in %s: %s", page_num, pdf_path, e)
    except fitz.FileDataError as e:
        logger.error("Corrupted PDF %s: %s", pdf_path, e)
    except Exception as e:
        logger.error("Error processing %s: %s", pdf_path, e)

    return paragraphs


def _extract_year_from_filename(filename: str) -> str:
    """Extract year from filenames like ``document_YYYY-MM-DD_XXX.pdf``."""
    match = re.search(r"document_(\d{4})-", filename)
    return match.group(1) if match else ""


def extract_text_from_pdfs(pdf_dir: str, output_csv: str, batch_size: int = 50) -> None:
    """
    Full OCR pipeline: iterate over PDFs, extract paragraphs, write CSV.

    This is the main entry point for **stage 1** of the pipeline.
    """
    logger.info("Starting PDF extraction from %s …", pdf_dir)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    counters = {
        "table_of_contents": 0, "acronyms": 0, "non_latin": 0,
        "short_paragraph": 0, "too_many_numbers": 0, "all_caps": 0,
        "references": 0, "unwanted_language": 0,
    }

    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

    # Write header
    with open(output_csv, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(["filename", "year", "text"])

    # Process in batches
    for i in range(0, len(pdf_files), batch_size):
        batch = pdf_files[i : i + batch_size]
        with tqdm(batch, desc=f"Batch {i // batch_size + 1}") as pbar:
            for pdf_file in pbar:
                pdf_path = os.path.join(pdf_dir, pdf_file)
                try:
                    year = _extract_year_from_filename(pdf_file)
                    paragraphs = _extract_paragraphs_from_pdf(pdf_path, counters)
                    if paragraphs:
                        with open(output_csv, "a", newline="", encoding="utf-8") as fh:
                            writer = csv.writer(fh)
                            for p in paragraphs:
                                writer.writerow([pdf_file, year, p])
                except Exception as e:
                    logger.error("Error processing %s: %s", pdf_path, e)

    logger.info("PDF extraction complete.")
    for reason, count in counters.items():
        logger.info("  Skipped (%s): %d", reason, count)

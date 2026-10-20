"""
Central configuration for the v2 pipeline.

Everything that changes results lives here so a run can be described by
this file plus the commit hash. Paths can be overridden with environment
variables (see .env.example) or CLI flags.
"""

import os
from datetime import date
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is optional
    pass

PROJECT_ROOT = Path(__file__).resolve().parent

# ---------- Paths ----------
PDF_DIR = Path(os.getenv("PDF_DIR", PROJECT_ROOT / "data" / "pdfs"))
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "outputs"))

# Optional DEC metadata export (the CSV from the USAID librarian).
# Used for document type / country controls. Leave unset if unavailable.
METADATA_CSV = os.getenv("METADATA_CSV")  # e.g. data/dec_metadata.csv
# Column names in that CSV. `None` = try to auto-detect (see preprocessing.load_metadata).
META_FILENAME_COL = os.getenv("META_FILENAME_COL")
META_DOCTYPE_COL = os.getenv("META_DOCTYPE_COL")
META_COUNTRY_COL = os.getenv("META_COUNTRY_COL")

# ---------- Administration windows ----------
TRUMP_START = date(2017, 1, 20)
BIDEN_START = date(2021, 1, 20)
BIDEN_END = date(2025, 1, 20)

# Robustness: drop documents dated within this many days after each
# inauguration (documents early in a term were mostly designed under the
# predecessor). Used only in the lag-robustness analysis.
LAG_EXCLUSION_DAYS = 365

# ---------- Extraction ----------
MIN_PARAGRAPH_WORDS = 10
# Paragraphs longer than this are split at sentence boundaries. ~180 words
# is ~230-270 word-piece tokens for policy English, under all-mpnet-base-v2's
# 384-token limit. 02_embed.py reports the share of chunks that still truncate.
MAX_CHUNK_WORDS = 180
# A block that repeats (after normalising digits) on at least this share of
# a document's pages is treated as a running header/footer and dropped.
HEADER_FOOTER_MIN_PAGE_SHARE = 0.3
LANGDETECT_SEED = 0
# Drop a paragraph only if langdetect is at least this confident it is not English.
NON_ENGLISH_MIN_PROB = 0.90

# ---------- Embedding ----------
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
EMBED_BATCH_SIZE = 64
EMBED_SHARD_SIZE = 20_000       # chunks per saved shard; lets 02_embed.py stop and resume
EMBED_FP16 = True               # half precision on Apple Silicon / CUDA: ~2x faster, negligible effect

# ---------- Topic model ----------
USE_SEED_TOPICS = True          # set False (or --no-seeds) for the unseeded robustness model
UMAP_N_NEIGHBORS = 15
UMAP_N_COMPONENTS = 5
UMAP_MIN_DIST = 0.0             # BERTopic default; tighter packing helps clustering
UMAP_METRIC = "cosine"
KMEANS_N_INIT = 10
# UMAP is fit on a random sample and then applied to every chunk. Fitting on all
# ~380k chunks with a fixed seed runs single-threaded and takes hours; 100k is
# plenty to learn the manifold. Set to None to fit on everything.
UMAP_FIT_SAMPLE = 100_000

# Stability sweep: pick K from the sweep *before* looking at administration differences.
SWEEP_K_GRID = [20, 30, 40, 50, 60, 75]
SWEEP_SEEDS = [0, 1, 2]
FINAL_SEED = 0

VECTORIZER_NGRAM_RANGE = (1, 2)
VECTORIZER_MIN_DF = 20
VECTORIZER_MAX_DF = 0.95
# Tokens must start with a letter: drops bare numbers but keeps "covid-19", "sars-cov-2".
VECTORIZER_TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]*[a-zA-Z0-9]\b"
N_TOP_WORDS = 10

# ---------- LLM labels (optional, display only) ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# ---------- Analysis ----------
N_PERMUTATIONS = 10_000
FDR_ALPHA = 0.05
SINGLE_DOC_DOMINANCE = 0.5      # flag topics where one document supplies >50% of chunks
TOP_COUNTRIES = 20              # other countries collapse to "Other" in regressions
READING_SAMPLE_SIZE = 15        # chunks per topic exported for manual reading
SEPARABILITY_PERMUTATIONS = 200
RANDOM_SEED = 42

# Doc-level mention tracker: share of documents mentioning each term family.
MENTION_TERMS = {
    "covid": r"\bcovid|\bsars-cov-2|\bcoronavirus",
    "antimicrobial_resistance": r"antimicrobial resistance|\bamr\b",
    "family_planning": r"family planning|contracepti",
    "mexico_city_plgha": r"mexico city policy|protecting life in global health|\bplgha\b",
    "multilateral": r"\bmultilateral",
    "who": r"world health organization",
    "global_fund": r"global fund",
    "gavi": r"\bgavi\b",
    "unfpa": r"\bunfpa\b",
    "health_security": r"global health security|\bghsa\b|pandemic preparedness",
    "climate": r"climate change|climate resilien",
    "equity": r"health equity|\bequitable\b",
    "localization": r"locali[sz]ation|local partner|locally led",
}

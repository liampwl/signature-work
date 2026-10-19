"""
Central configuration for the USAID policy analysis pipeline.

All paths and settings are configurable via environment variables or CLI arguments.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------- Paths ----------
# Override these via environment variables or pass via argparse in scripts
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PDF_DIR = os.getenv("PDF_DIR", str(PROJECT_ROOT / "data" / "pdfs"))
DEFAULT_CSV_PATH = os.getenv("CSV_PATH", str(PROJECT_ROOT / "data" / "processed_paragraphs.csv"))
DEFAULT_MODEL_DIR = os.getenv("MODEL_DIR", str(PROJECT_ROOT / "models"))
DEFAULT_OUTPUT_DIR = os.getenv("OUTPUT_DIR", str(PROJECT_ROOT / "outputs"))

# ---------- API ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_DELAY = int(os.getenv("OPENAI_DELAY", "10"))

# ---------- Model defaults ----------
UMAP_N_NEIGHBORS = 15
UMAP_N_COMPONENTS = 5
UMAP_MIN_DIST = 0.1
UMAP_SPREAD = 1.5

HDBSCAN_MIN_CLUSTER_SIZE = 50
HDBSCAN_MIN_SAMPLES = 5

KMEANS_N_CLUSTERS = 50

VECTORIZER_NGRAM_RANGE = (1, 3)
VECTORIZER_MIN_DF = 20
VECTORIZER_MAX_DF = 0.95

MIN_TOPIC_SIZE = 50
NR_TOPICS = "auto"

EMBEDDING_MODEL_NAME = "paraphrase-MiniLM-L6-v2"

# ---------- Analysis defaults ----------
NULL_MODEL_PERMUTATIONS = 100
SIMILARITY_SAMPLE_SIZE = 2000
COHERENCE_DOC_LIMIT = 5000
TOPICS_OVER_TIME_BINS = 20
FOCUSING_EVENT_WINDOW = 60

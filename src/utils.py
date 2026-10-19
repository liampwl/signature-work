"""Shared utilities."""

from __future__ import annotations

import json
import logging
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(name)


def load_corpus(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load chunks.csv and documents.csv written by stage 1."""
    chunks = pd.read_csv(data_dir / "chunks.csv", low_memory=False)
    docs = pd.read_csv(data_dir / "documents.csv", dtype={"doc_type": str, "country": str})
    docs["date"] = pd.to_datetime(docs["date"], errors="coerce")
    return chunks, docs


def write_run_info(out_dir: Path, extra: dict | None = None) -> None:
    """Record config + versions + git commit next to every output folder."""
    import config as cfg

    info = {
        k: (str(v) if not isinstance(v, (int, float, bool, list, dict, type(None))) else v)
        for k, v in vars(cfg).items()
        if k.isupper() and "KEY" not in k
    }
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True
        ).strip()
    except Exception:
        info["git_commit"] = None
    info["python"] = platform.python_version()
    for pkg in ["bertopic", "umap", "sklearn", "sentence_transformers", "pymupdf", "numpy", "pandas"]:
        try:
            info[f"version_{pkg}"] = __import__(pkg).__version__
        except Exception:
            pass
    if extra:
        info.update(extra)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2, default=str))


def doc_topic_shares(chunks: pd.DataFrame, topics: np.ndarray, doc_ids: np.ndarray, n_topics: int) -> np.ndarray:
    """
    Return an (n_docs x n_topics) matrix: for each document, the share of its
    chunks assigned to each topic. Rows follow the order of `doc_ids`.
    """
    idx = {d: i for i, d in enumerate(doc_ids)}
    mat = np.zeros((len(doc_ids), n_topics))
    for d, t in zip(chunks["doc_id"].to_numpy(), topics):
        i = idx.get(d)
        if i is not None:
            mat[i, t] += 1
    totals = mat.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1
    return mat / totals

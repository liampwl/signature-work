"""
Stage 2: embed chunks once and cache them.

Every later step (sweep, final model, separability test) reuses the cached
matrix, so model selection doesn't re-embed the corpus each time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
from src.utils import get_logger

logger = get_logger(__name__)


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_embedding_model(name: str = cfg.EMBEDDING_MODEL_NAME):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, device=_device())


def truncation_report(model, texts: list[str]) -> pd.Series:
    """Token counts per chunk, so we can report how much text the model actually sees."""
    tok = model.tokenizer
    lengths = [len(ids) for ids in tok(texts, add_special_tokens=True, truncation=False)["input_ids"]]
    return pd.Series(lengths, name="n_tokens")


def embed_chunks(data_dir: Path, model_name: str = cfg.EMBEDDING_MODEL_NAME) -> None:
    chunks = pd.read_csv(data_dir / "chunks.csv", low_memory=False)
    texts = chunks["text"].astype(str).tolist()
    model = load_embedding_model(model_name)
    max_len = model.max_seq_length
    logger.info("Model %s | device %s | max_seq_length %d", model_name, model.device, max_len)

    lengths = truncation_report(model, texts)
    over = (lengths > max_len).mean()
    logger.info(
        "Token length: median %d, p95 %d, max %d | %.2f%% of chunks exceed %d and will truncate",
        lengths.median(), lengths.quantile(0.95), lengths.max(), 100 * over, max_len,
    )
    pd.DataFrame({"chunk_id": chunks["chunk_id"], "n_tokens": lengths}).to_csv(
        data_dir / "token_lengths.csv", index=False
    )

    emb = model.encode(
        texts,
        batch_size=cfg.EMBED_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    np.save(data_dir / "embeddings.npy", emb)
    (data_dir / "embeddings_model.txt").write_text(f"{model_name}\nmax_seq_length={max_len}\n")
    logger.info("Saved embeddings %s -> %s", emb.shape, data_dir / "embeddings.npy")


def load_embeddings(data_dir: Path, n_expected: int) -> np.ndarray:
    emb = np.load(data_dir / "embeddings.npy")
    if emb.shape[0] != n_expected:
        raise ValueError(
            f"embeddings.npy has {emb.shape[0]} rows but chunks.csv has {n_expected}. Re-run 02_embed.py."
        )
    return emb

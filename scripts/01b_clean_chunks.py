#!/usr/bin/env python3
"""
Stage 1b: remove boilerplate / extraction-debris chunks from an EXISTING
chunks.csv + embeddings.npy, keeping rows aligned, so nothing is re-embedded.

New extractions already apply the same rules (preprocessing.chunk_flags);
this is for corpora extracted before those rules existed. Originals are kept
as chunks_unfiltered.csv / embeddings_unfiltered.npy, and the UMAP cache is
moved aside because it no longer matches.

    python scripts/01b_clean_chunks.py --data-dir data
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import _common  # noqa: F401
import config as cfg
from src.preprocessing import chunk_flags


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=cfg.DATA_DIR)
    args = ap.parse_args()
    d = args.data_dir

    raw_chunks, raw_emb = d / "chunks_unfiltered.csv", d / "embeddings_unfiltered.npy"
    if not raw_chunks.exists():
        shutil.copy(d / "chunks.csv", raw_chunks)
        shutil.copy(d / "embeddings.npy", raw_emb)
    chunks = pd.read_csv(raw_chunks, low_memory=False)
    emb = np.load(raw_emb, mmap_mode="r")
    assert len(chunks) == emb.shape[0], "chunks and embeddings are out of sync"

    flags = chunk_flags(chunks)
    keep = ~flags.any(axis=1).to_numpy()
    counts = flags.sum()
    counts["any"] = int((~keep).sum())
    counts.rename("chunks_removed").to_csv(d / "chunk_filter_counts.csv")
    print(counts.to_string())

    out = chunks[keep].reset_index(drop=True)
    out["chunk_id"] = range(len(out))
    out.to_csv(d / "chunks.csv", index=False)
    np.save(d / "embeddings.npy", np.ascontiguousarray(emb[keep]))

    if (d / "umap_cache").exists():
        dest = d / "umap_cache_unfiltered"
        if dest.exists():
            shutil.rmtree(dest)
        (d / "umap_cache").rename(dest)
    print(f"Kept {keep.sum()} of {len(keep)} chunks ({keep.mean():.1%}).")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Stage 2: embed every chunk once -> data/embeddings.npy (+ token_lengths.csv)

    python scripts/02_embed.py --data-dir data
"""
import argparse
from pathlib import Path

import _common  # noqa: F401
import config as cfg
from src.embedding import embed_chunks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=cfg.DATA_DIR)
    ap.add_argument("--model", default=cfg.EMBEDDING_MODEL_NAME)
    args = ap.parse_args()
    embed_chunks(args.data_dir, args.model)


if __name__ == "__main__":
    main()

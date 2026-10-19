#!/usr/bin/env python3
"""
Stage 3: sweep K x seeds and report stability / coherence / diversity.

Choose K from outputs/sweep/sweep_summary.csv BEFORE running stage 5, and
record the reason (e.g. "highest seed ARI among K with < 2 single-document
topics"). None of these criteria look at administration differences.

    python scripts/03_sweep.py --data-dir data --out-dir outputs/sweep [--no-seeds]
"""
import argparse
from pathlib import Path

import _common  # noqa: F401
import config as cfg
from src.embedding import load_embeddings
from src.sweep import run_sweep
from src.utils import load_corpus, write_run_info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=cfg.DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=cfg.OUTPUT_DIR / "sweep")
    ap.add_argument("--no-seeds", action="store_true", help="unseeded model (robustness)")
    ap.add_argument("--k", type=int, nargs="+", default=cfg.SWEEP_K_GRID)
    ap.add_argument("--seeds", type=int, nargs="+", default=cfg.SWEEP_SEEDS)
    args = ap.parse_args()

    chunks, _ = load_corpus(args.data_dir)
    emb = load_embeddings(args.data_dir, len(chunks))
    seeded = cfg.USE_SEED_TOPICS and not args.no_seeds
    out = args.out_dir / ("seeded" if seeded else "unseeded")
    write_run_info(out, {"stage": "sweep", "seeded": seeded, "k_grid": args.k, "seeds": args.seeds})
    run_sweep(args.data_dir, out, chunks, emb, seeded, args.k, args.seeds)


if __name__ == "__main__":
    main()

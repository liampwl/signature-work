#!/usr/bin/env python3
"""
Stage 4: fit the final model at the chosen K and save assignments + topic info.

    python scripts/04_train_model.py --data-dir data --k 40 --out-dir outputs/final_k40 [--no-seeds] [--gpt-labels]
"""
import argparse
from pathlib import Path

import _common  # noqa: F401
import config as cfg
from src.embedding import load_embedding_model, load_embeddings
from src.modeling import TermMatrices, cluster, fit_representations, get_reduced, gpt_labels, save_final
from src.utils import get_logger, load_corpus, write_run_info

logger = get_logger(__name__)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=cfg.DATA_DIR)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--seed", type=int, default=cfg.FINAL_SEED)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--no-seeds", action="store_true")
    ap.add_argument("--gpt-labels", action="store_true", help="add display-only GPT labels (needs OPENAI_API_KEY)")
    ap.add_argument("--k-rationale", default="", help="one line on why this K was chosen from the sweep")
    args = ap.parse_args()

    seeded = cfg.USE_SEED_TOPICS and not args.no_seeds
    out = args.out_dir or cfg.OUTPUT_DIR / f"final_k{args.k}_{'seeded' if seeded else 'unseeded'}"
    chunks, docs = load_corpus(args.data_dir)
    emb = load_embeddings(args.data_dir, len(chunks))
    model = load_embedding_model()

    reduced = get_reduced(args.data_dir, emb, args.seed, seeded, model)
    labels = cluster(reduced, args.k, args.seed)

    texts = chunks["text"].astype(str).tolist()
    topic_model = fit_representations(texts, emb, labels, model)
    gpt = gpt_labels(topic_model) if args.gpt_labels else {}
    terms = TermMatrices(texts)
    info = save_final(out, topic_model, labels, chunks, docs, terms, args.k, gpt)
    write_run_info(out, {"stage": "final", "k": args.k, "seed": args.seed, "seeded": seeded,
                         "k_rationale": args.k_rationale})
    logger.info("Saved final model to %s\n%s", out,
                info[["topic", "n_chunks", "n_docs", "top_doc_share", "npmi", "keywords"]].to_string(index=False))


if __name__ == "__main__":
    main()

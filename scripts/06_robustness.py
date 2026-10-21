#!/usr/bin/env python3
"""
Stage 6: do the primary model's findings survive other K, other seeds, and no seed topics?

Topics are matched across runs by chunk overlap: every run assigns the same
chunks, so for each primary topic the best match in another run is the topic
sharing the largest Jaccard overlap of chunks. A finding "replicates" in a run
when the matched topic (Jaccard >= --min-jaccard) moves in the same direction
with q < 0.05.

    python scripts/06_robustness.py --primary outputs/final_k40_seeded \
        --runs outputs/final_k30_seeded outputs/final_k60_seeded outputs/final_k40_seed1 ...
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import _common  # noqa: F401
import config as cfg


def load(run: Path):
    topics = pd.read_csv(run / "topics.csv")["topic"].to_numpy()
    comp = pd.read_csv(run / "analysis" / "administration_comparison.csv").set_index("topic")
    return topics, comp


def best_matches(a: np.ndarray, b: np.ndarray) -> pd.DataFrame:
    ct = pd.crosstab(a, b)
    size_a, size_b = ct.sum(axis=1), ct.sum(axis=0)
    jac = ct / (size_a.to_numpy()[:, None] + size_b.to_numpy()[None, :] - ct)
    return pd.DataFrame({"match": jac.idxmax(axis=1), "jaccard": jac.max(axis=1)})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--primary", type=Path, required=True)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--min-jaccard", type=float, default=0.3)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    p_topics, p_comp = load(args.primary)
    table = p_comp[[c for c in ["keywords", "n_docs", "diff_biden_minus_trump", "q_perm", "q_trend"] if c in p_comp]].copy()
    sign = np.sign(table["diff_biden_minus_trump"])
    rep_perm = pd.Series(0, index=table.index)
    rep_trend = pd.Series(0, index=table.index)

    for run in args.runs:
        r_topics, r_comp = load(run)
        assert len(r_topics) == len(p_topics), f"{run} was fit on a different corpus"
        m = best_matches(p_topics, r_topics).reindex(table.index)
        name = run.name
        table[f"{name}_jaccard"] = m["jaccard"].round(2)
        matched = r_comp.reindex(m["match"].to_numpy()).set_index(m.index)
        table[f"{name}_diff"] = matched["diff_biden_minus_trump"]
        good = m["jaccard"] >= args.min_jaccard
        same = np.sign(matched["diff_biden_minus_trump"]) == sign
        rep_perm += (good & same & (matched["q_perm"] < cfg.FDR_ALPHA)).astype(int)
        if "q_trend" in matched:
            table[f"{name}_q_trend"] = matched["q_trend"]
            rep_trend += (good & same & (matched["q_trend"] < cfg.FDR_ALPHA)).astype(int)

    n = len(args.runs)
    table.insert(2, f"replicates_perm_of_{n}", rep_perm)
    table.insert(3, f"replicates_trend_of_{n}", rep_trend)
    table = table.sort_values("diff_biden_minus_trump", key=np.abs, ascending=False)
    out = args.out or args.primary / "analysis" / "robustness.csv"
    table.to_csv(out)
    show = table[[c for c in table.columns if not c.endswith(("_jaccard", "_diff", "_q_trend"))]].copy()
    show["keywords"] = show["keywords"].str.slice(0, 45)
    print(show.head(25).round(4).to_string())
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()

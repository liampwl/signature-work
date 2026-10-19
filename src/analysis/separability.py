"""
How distinguishable are Trump-era and Biden-era documents?

Replaces v1's mean pairwise cosine similarity, which has no null distribution
and cannot detect a shift in the *mix* of content. Here a cross-validated
logistic regression tries to predict administration from each document, and
the AUC is compared with a label-permutation null.

AUC ~0.5 means the two periods look alike; a high AUC means they differ.
A high AUC does not by itself mean ideology: time trends (COVID) and changes
in document-type mix also make periods separable. Read it together with the
covariate regressions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

import config as cfg
from src.utils import get_logger

logger = get_logger(__name__)


def doc_mean_embeddings(embeddings: np.ndarray, chunk_doc_ids: np.ndarray, doc_ids: np.ndarray) -> np.ndarray:
    idx = {d: i for i, d in enumerate(doc_ids)}
    out = np.zeros((len(doc_ids), embeddings.shape[1]), dtype=np.float64)
    counts = np.zeros(len(doc_ids))
    for row, d in enumerate(chunk_doc_ids):
        i = idx.get(d)
        if i is not None:
            out[i] += embeddings[row]
            counts[i] += 1
    out /= np.maximum(counts, 1)[:, None]
    return out


def _cv_auc(X: np.ndarray, y: np.ndarray, seed: int) -> float:
    clf = LogisticRegression(C=1.0, max_iter=2000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    prob = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    return roc_auc_score(y, prob)


def separability(features: dict[str, np.ndarray], is_biden: np.ndarray, out_dir: Path,
                 n_perm: int = cfg.SEPARABILITY_PERMUTATIONS) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    y = is_biden.astype(int)
    rows = []
    for name, X in features.items():
        auc = _cv_auc(X, y, cfg.RANDOM_SEED)
        null = np.array([_cv_auc(X, rng.permutation(y), cfg.RANDOM_SEED) for _ in range(n_perm)])
        rows.append({
            "features": name, "cv_auc": auc,
            "null_auc_mean": null.mean(), "null_auc_95th": np.quantile(null, 0.95),
            "p_value": (np.sum(null >= auc) + 1) / (n_perm + 1),
        })
        logger.info("Separability [%s]: AUC %.3f (null mean %.3f)", name, auc, null.mean())
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "separability.csv", index=False)
    return out

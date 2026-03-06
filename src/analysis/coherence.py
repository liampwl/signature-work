"""
Topic coherence metrics (c_v and u_mass) to assess model quality.
"""

from __future__ import annotations

import os

import nltk
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm

import config as cfg
from src.utils import get_logger

logger = get_logger(__name__)


def calculate_topic_coherence(topic_model, documents, output_dir):
    """
    Calculate c_v and u_mass coherence for every topic in the model.

    Returns a DataFrame with per-topic scores.
    """
    logger.info("Calculating topic coherence metrics ...")

    # Ensure NLTK resources exist
    for resource, name in [("tokenizers/punkt", "punkt"), ("corpora/stopwords", "stopwords")]:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(name)

    stop_words = set(stopwords.words("english"))

    logger.info("Tokenizing documents (limit=%d) ...", cfg.COHERENCE_DOC_LIMIT)
    tokenized: list[list[str]] = []
    for doc in tqdm(documents[: cfg.COHERENCE_DOC_LIMIT], desc="Tokenizing"):
        tokens = word_tokenize(doc.lower())
        tokenized.append([t for t in tokens if t.isalpha() and t not in stop_words and len(t) > 2])

    dictionary = Dictionary(tokenized)

    # Build keyword lists per topic
    topic_info = topic_model.get_topic_info()
    topics_keywords: dict[int, list[str]] = {}

    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        if tid == -1:
            continue
        name = row["Name"]
        if name and "_" in name:
            parts = name.split("_", 1)
            if len(parts) > 1:
                words = [w.lower() for w in parts[1].split() if w.lower() in dictionary.token2id]
                if len(words) >= 2:
                    topics_keywords[tid] = words

    # Calculate coherence
    measures = ["c_v", "u_mass"]
    rows: list[dict] = []

    for measure in measures:
        logger.info("Computing %s coherence ...", measure)
        for tid, keywords in tqdm(topics_keywords.items(), desc=measure):
            filtered = [w for w in keywords if w in dictionary.token2id]
            if len(filtered) < 2:
                score = 0.01
            else:
                try:
                    cm = CoherenceModel(
                        topics=[filtered], texts=tokenized,
                        dictionary=dictionary, coherence=measure,
                    )
                    score = cm.get_coherence()
                except Exception as exc:
                    logger.warning("Coherence error for topic %d: %s", tid, exc)
                    score = np.nan

            topic_name = topic_info.loc[topic_info["Topic"] == tid, "Name"].values
            rows.append({
                "Topic": tid,
                "Name": topic_name[0] if len(topic_name) else f"Topic {tid}",
                "Coherence_Measure": measure,
                "Score": score,
                "Num_Keywords": len(filtered),
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "topic_coherence_metrics.csv"), index=False)

    avg = df.groupby("Coherence_Measure")["Score"].agg(["mean", "median", "std"]).reset_index()
    avg.to_csv(os.path.join(output_dir, "average_topic_coherence.csv"), index=False)

    cv_mean = avg.loc[avg["Coherence_Measure"] == "c_v", "mean"].values
    if len(cv_mean):
        logger.info("Average c_v coherence: %.4f", cv_mean[0])

    return df

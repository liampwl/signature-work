"""
Document similarity analysis: embedding-space visualization and
within- vs. cross-administration cosine similarity.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

import config as cfg
from src.utils import get_logger

logger = get_logger(__name__)


def analyze_document_similarity(
    topic_model, documents, administrations, output_dir,
    sample_size: int | None = None,
):
    """
    Compute t-SNE embeddings and cosine similarity within / across
    administrations.  Sampling is applied when len(documents) > sample_size.
    """
    sample_size = sample_size or cfg.SIMILARITY_SAMPLE_SIZE
    logger.info("Analyzing document similarity (sample_size=%d) ...", sample_size)

    if len(documents) > sample_size:
        trump_idx = [i for i, a in enumerate(administrations) if a == "Trump"]
        biden_idx = [i for i, a in enumerate(administrations) if a == "Biden"]

        np.random.seed(42)
        s_trump = np.random.choice(trump_idx, min(len(trump_idx), sample_size // 2), replace=False)
        s_biden = np.random.choice(biden_idx, min(len(biden_idx), sample_size // 2), replace=False)
        sampled = np.concatenate([s_trump, s_biden])

        docs = [documents[i] for i in sampled]
        admins = [administrations[i] for i in sampled]
    else:
        docs = documents
        admins = administrations
        sampled = list(range(len(documents)))

    # Embeddings
    logger.info("Extracting document embeddings ...")
    embeddings = topic_model._extract_embeddings(docs, method="document", verbose=True)

    # t-SNE
    logger.info("Computing t-SNE projection ...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    proj = tsne.fit_transform(embeddings)

    emb_df = pd.DataFrame({
        "x": proj[:, 0], "y": proj[:, 1],
        "Administration": admins, "Original_Index": sampled,
    })
    emb_df.to_csv(os.path.join(output_dir, "document_embeddings.csv"), index=False)

    fig = px.scatter(
        emb_df, x="x", y="y", color="Administration",
        color_discrete_map={"Trump": "red", "Biden": "blue"},
        title="Document Embedding Space by Administration",
        opacity=0.7, width=1000, height=800,
    )
    fig.write_html(os.path.join(output_dir, "document_embedding_scatter.html"))

    # Cosine similarity
    logger.info("Computing cosine similarity ...")
    trump_emb = embeddings[np.array(admins) == "Trump"]
    biden_emb = embeddings[np.array(admins) == "Biden"]

    if len(trump_emb) and len(biden_emb):
        t_sim = cosine_similarity(trump_emb)
        np.fill_diagonal(t_sim, 0)
        b_sim = cosine_similarity(biden_emb)
        np.fill_diagonal(b_sim, 0)
        cross = cosine_similarity(trump_emb, biden_emb)

        summary = pd.DataFrame([{
            "Within_Trump": t_sim.mean(),
            "Within_Biden": b_sim.mean(),
            "Between_Administrations": cross.mean(),
            "Trump_Documents": len(trump_emb),
            "Biden_Documents": len(biden_emb),
        }])
        summary.to_csv(os.path.join(output_dir, "administration_similarity.csv"), index=False)

        sim_data = pd.DataFrame({
            "Comparison": ["Within Trump", "Within Biden", "Between Administrations"],
            "Average Similarity": [t_sim.mean(), b_sim.mean(), cross.mean()],
        })
        fig = px.bar(
            sim_data, x="Comparison", y="Average Similarity",
            title="Average Document Similarity Within and Across Administrations",
            width=800, height=500,
        )
        fig.write_html(os.path.join(output_dir, "administration_similarity_chart.html"))

        logger.info(
            "Similarity — Trump: %.4f, Biden: %.4f, Cross: %.4f",
            t_sim.mean(), b_sim.mean(), cross.mean(),
        )

    return emb_df

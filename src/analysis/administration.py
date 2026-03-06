"""
Administration-level analyses: topic distribution comparison and null-model
permutation testing.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from tqdm import tqdm

from src.utils import get_logger, get_topic_labels

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Topic distribution comparison
# ---------------------------------------------------------------------------

def analyze_administration_differences(topic_model, documents, administrations, output_dir):
    """
    Compare topic prevalence between Trump and Biden administrations and
    create a grouped bar chart of the top 15 most divergent topics.
    """
    logger.info("Analyzing topic distribution differences between administrations ...")

    topics, _ = topic_model.transform(documents)

    trump_docs = [d for d, a in zip(documents, administrations) if a == "Trump"]
    biden_docs = [d for d, a in zip(documents, administrations) if a == "Biden"]

    trump_topics, _ = topic_model.transform(trump_docs)
    biden_topics, _ = topic_model.transform(biden_docs)

    trump_freq = pd.Series(trump_topics).value_counts(normalize=True)
    biden_freq = pd.Series(biden_topics).value_counts(normalize=True)

    comparison = pd.DataFrame({"Trump": trump_freq, "Biden": biden_freq}).fillna(0)

    labels = get_topic_labels(topic_model)
    comparison["Label"] = comparison.index.map(lambda x: labels.get(x, f"Topic {x}"))
    comparison["Difference"] = (comparison["Trump"] - comparison["Biden"]).abs()
    comparison = comparison.sort_values("Difference", ascending=False)

    comparison.to_csv(os.path.join(output_dir, "administration_topic_comparison.csv"))

    # Visualization — top 15
    top = comparison.head(15)
    fig = px.bar(
        top, x="Label", y=["Trump", "Biden"], barmode="group",
        labels={"value": "Topic Prevalence", "variable": "Administration"},
        color_discrete_map={"Trump": "red", "Biden": "blue"},
        title="Top 15 Topics with Greatest Shift Between Trump and Biden Administrations",
        height=600, width=1000,
    )
    fig.write_html(os.path.join(output_dir, "administration_topic_shift.html"))

    logger.info("Saved administration comparison → %s", output_dir)
    return comparison


# ---------------------------------------------------------------------------
# Null-model permutation test
# ---------------------------------------------------------------------------

def run_null_model_comparison(
    topic_model, documents, administrations, output_dir,
    n_permutations: int = 100,
):
    """
    Compare actual topic distributions against a null model where
    administration labels are randomly shuffled (*n_permutations* times).
    """
    logger.info("Running null model comparison (%d permutations) ...", n_permutations)

    topics, _ = topic_model.transform(documents)

    trump_docs = [d for d, a in zip(documents, administrations) if a == "Trump"]
    biden_docs = [d for d, a in zip(documents, administrations) if a == "Biden"]

    trump_topics, _ = topic_model.transform(trump_docs)
    biden_topics, _ = topic_model.transform(biden_docs)

    actual_trump = pd.Series(trump_topics).value_counts(normalize=True)
    actual_biden = pd.Series(biden_topics).value_counts(normalize=True)

    all_topics = sorted(set(list(actual_trump.index) + list(actual_biden.index)))

    actual_diffs = {
        t: abs(actual_trump.get(t, 0) - actual_biden.get(t, 0))
        for t in all_topics
    }

    null_diffs: dict[int, list[float]] = {t: [] for t in all_topics}

    for _ in tqdm(range(n_permutations), desc="Permutations"):
        shuffled = np.random.permutation(administrations)
        s_trump = [d for d, a in zip(documents, shuffled) if a == "Trump"]
        s_biden = [d for d, a in zip(documents, shuffled) if a == "Biden"]

        if not s_trump or not s_biden:
            continue

        st, _ = topic_model.transform(s_trump)
        sb, _ = topic_model.transform(s_biden)
        st_dist = pd.Series(st).value_counts(normalize=True)
        sb_dist = pd.Series(sb).value_counts(normalize=True)

        for t in all_topics:
            null_diffs[t].append(abs(st_dist.get(t, 0) - sb_dist.get(t, 0)))

    # p-values
    p_values = {}
    for t in all_topics:
        arr = np.array(null_diffs[t])
        p_values[t] = (np.sum(arr >= actual_diffs[t]) + 1) / (len(arr) + 1)

    labels = get_topic_labels(topic_model)
    rows = [
        {
            "Topic": t,
            "Name": labels.get(t, f"Topic {t}"),
            "Trump_Frequency": actual_trump.get(t, 0),
            "Biden_Frequency": actual_biden.get(t, 0),
            "Abs_Difference": actual_diffs[t],
            "p_value": p_values[t],
            "Significant": p_values[t] < 0.05,
        }
        for t in all_topics
    ]

    results = pd.DataFrame(rows).sort_values("p_value")
    results.to_csv(os.path.join(output_dir, "null_model_comparison.csv"), index=False)

    significant = results[results["Significant"]]
    if not significant.empty:
        plt.figure(figsize=(12, 8))
        plt.bar(
            range(len(significant)),
            significant["Abs_Difference"],
            tick_label=[
                f"{r['Topic']}: {r['Name'][:30]}..."
                if len(r["Name"]) > 30
                else f"{r['Topic']}: {r['Name']}"
                for _, r in significant.iterrows()
            ],
        )
        plt.title("Significant Topic Differences Between Administrations")
        plt.ylabel("Absolute Difference in Topic Frequency")
        plt.xlabel("Topics")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "significant_topic_differences.png"), dpi=300)
        plt.close()

    logger.info("Null model comparison complete — %d significant topics.", len(significant))
    return results

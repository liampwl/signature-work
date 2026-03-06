"""
Temporal analyses: granularity sweeps, topic evolution trends, and
policy attention shift measurement.
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px

from src.utils import get_logger, get_topic_labels

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Multi-granularity topics over time
# ---------------------------------------------------------------------------

def analyze_temporal_granularity(topic_model, documents, timestamps, output_dir):
    """
    Compute topics-over-time at quarterly and yearly granularity and
    produce a heatmap of the top 15 topics.
    """
    logger.info("Analyzing topics at different temporal granularities ...")

    dates = [datetime(year=int(ts), month=1, day=1) for ts in timestamps]
    granularities = {"quarterly": 4, "yearly": 1}
    results = {}

    year_span = max(timestamps) - min(timestamps) + 1

    for label, multiplier in granularities.items():
        logger.info("Processing %s granularity ...", label)
        nr_bins = multiplier * year_span

        tot = topic_model.topics_over_time(
            documents, dates, nr_bins=nr_bins,
            evolution_tuning=True, global_tuning=True,
        )
        results[label] = tot
        tot.to_csv(os.path.join(output_dir, f"topics_over_time_{label}.csv"), index=False)

        fig = topic_model.visualize_topics_over_time(
            tot, top_n_topics=10, width=1200, height=800,
        )
        fig.write_html(os.path.join(output_dir, f"topics_over_time_{label}.html"))

    # Heatmap from yearly data
    logger.info("Creating topic evolution heatmap ...")
    tot = results["yearly"]
    labels = get_topic_labels(topic_model)
    tot["TopicName"] = tot["Topic"].map(lambda x: labels.get(x, f"Topic {x}"))

    top_topics = tot.groupby("Topic")["Frequency"].sum().nlargest(15).index.tolist()
    filtered = tot[tot["Topic"].isin(top_topics)]

    heatmap = filtered.pivot_table(
        index="TopicName", columns="Timestamp", values="Frequency", aggfunc="mean",
    ).fillna(0)

    fig = px.imshow(
        heatmap,
        labels=dict(x="Time", y="Topic", color="Frequency"),
        color_continuous_scale="Viridis",
        title="Topic Evolution Over Time",
        height=800, width=1200,
    )
    fig.write_html(os.path.join(output_dir, "topic_evolution_heatmap.html"))
    logger.info("Temporal analysis complete.")

    return results


# ---------------------------------------------------------------------------
# Topic evolution (increasing / decreasing)
# ---------------------------------------------------------------------------

def analyze_topic_evolution(topic_model, topics_over_time, output_dir):
    """
    Identify the 10 most-increasing and 10 most-decreasing topics.
    """
    logger.info("Analyzing topic evolution patterns ...")

    pivot = topics_over_time.pivot(index="Timestamp", columns="Topic", values="Frequency")
    changes = pivot.diff()
    total_change = changes.sum()

    labels = get_topic_labels(topic_model)

    def _make_df(series):
        return pd.DataFrame({
            "Topic": series.index,
            "Change_Value": series.values,
            "Name": [labels.get(t, f"Topic {t}") for t in series.index],
        })

    inc = _make_df(total_change.nlargest(10))
    dec = _make_df(total_change.nsmallest(10))

    inc.to_csv(os.path.join(output_dir, "increasing_topics.csv"), index=False)
    dec.to_csv(os.path.join(output_dir, "decreasing_topics.csv"), index=False)

    for df, direction in [(inc, "Increase"), (dec, "Decrease")]:
        fig = px.bar(
            df, x="Name", y="Change_Value",
            title=f"Topics with Greatest {direction} Over Time",
            labels={"Change_Value": "Total Change in Frequency", "Name": "Topic"},
            height=600, width=900,
        )
        fig.write_html(os.path.join(output_dir, f"{direction.lower()}ing_topics.html"))

    logger.info("Topic evolution analysis complete.")
    return inc, dec


# ---------------------------------------------------------------------------
# Policy attention shifts (normalized, Biden − Trump)
# ---------------------------------------------------------------------------

def analyze_policy_attention_shifts(topic_model, documents, timestamps, administrations, output_dir):
    """
    Compute normalized attention scores per topic per administration and
    visualize the largest shifts.
    """
    logger.info("Analyzing policy attention shifts ...")

    doc_data = pd.DataFrame({
        "text": documents, "timestamp": timestamps, "administration": administrations,
    })
    topics, _ = topic_model.transform(documents)
    doc_data["topic"] = topics

    admin_topics = doc_data.groupby(["administration", "topic"]).size().reset_index(name="count")
    admin_totals = admin_topics.groupby("administration")["count"].sum().reset_index(name="total")
    admin_topics = admin_topics.merge(admin_totals, on="administration")
    admin_topics["norm_attention"] = admin_topics["count"] / admin_topics["total"]

    pivot = admin_topics.pivot(index="topic", columns="administration", values="norm_attention").fillna(0)
    pivot["attention_shift"] = pivot["Biden"] - pivot["Trump"]

    labels = get_topic_labels(topic_model)
    pivot["Name"] = pivot.index.map(lambda x: labels.get(x, f"Topic {x}"))

    pivot.sort_values("attention_shift", ascending=False).to_csv(
        os.path.join(output_dir, "policy_attention_shifts.csv"),
    )

    top_shifts = pd.concat([
        pivot.nlargest(10, "attention_shift"),
        pivot.nsmallest(10, "attention_shift"),
    ])

    fig = px.bar(
        top_shifts.reset_index(), x="Name", y="attention_shift",
        title="Largest Policy Attention Shifts Between Administrations",
        labels={"attention_shift": "Shift in Normalized Attention (Biden - Trump)"},
        color="attention_shift", color_continuous_scale="RdBu", height=600,
    )
    fig.write_html(os.path.join(output_dir, "policy_attention_shifts.html"))

    return pivot

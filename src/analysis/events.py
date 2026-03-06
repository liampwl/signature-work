"""
Event detection: focusing event spikes and punctuated equilibrium
volatility visualizations.
"""

from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import config as cfg
from src.utils import get_logger, get_topic_labels

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Focusing events (z-score spike detection)
# ---------------------------------------------------------------------------

def detect_focusing_events(
    topic_model, documents, timestamps, output_dir,
    window_size: int | None = None,
):
    """
    Detect sudden spikes in topic prominence using rolling z-scores.

    Returns a DataFrame of spike dates with z-scores.
    """
    window_size = window_size or cfg.FOCUSING_EVENT_WINDOW
    logger.info("Detecting focusing events (window=%d) ...", window_size)

    df = pd.DataFrame({"text": documents, "timestamp": timestamps})
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(str) + "-01-01")
    df = df.sort_values("timestamp")

    topics, _ = topic_model.transform(documents)
    df["topic"] = topics

    topic_counts = (
        df.groupby([pd.Grouper(key="timestamp", freq="M"), "topic"])
        .size()
        .unstack()
        .fillna(0)
    )

    rolling_mean = topic_counts.rolling(window=window_size, min_periods=1).mean()
    rolling_std = topic_counts.rolling(window=window_size, min_periods=1).std()
    z_scores = (topic_counts - rolling_mean) / rolling_std.replace(0, 1)

    spikes = (z_scores > 2).astype(int)
    labels = get_topic_labels(topic_model)

    rows = []
    for topic in spikes.columns:
        for date in spikes.index[spikes[topic] == 1]:
            rows.append({
                "Date": date,
                "Topic": topic,
                "Topic_Name": labels.get(topic, f"Topic {topic}"),
                "Z_Score": float(z_scores.loc[date, topic]),
            })

    spike_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Date", "Topic", "Topic_Name", "Z_Score"])

    if not spike_df.empty:
        spike_df["Z_Score"] = spike_df["Z_Score"].astype(float)
        spike_df.sort_values("Z_Score", ascending=False).to_csv(
            os.path.join(output_dir, "potential_focusing_events.csv"), index=False,
        )

        fig = px.scatter(
            spike_df.nlargest(15, "Z_Score"),
            x="Date", y="Z_Score", color="Topic_Name", size="Z_Score",
            title="Potential Focusing Events (Sudden Topic Spikes)",
            labels={"Z_Score": "Intensity (Z-Score)"}, height=600,
        )
        fig.write_html(os.path.join(output_dir, "focusing_events.html"))

    return spike_df


# ---------------------------------------------------------------------------
# Punctuated equilibrium volatility
# ---------------------------------------------------------------------------

def visualize_punctuated_equilibrium(topic_model, topics_over_time, output_dir):
    """
    Visualize month-to-month volatility in topic attention with
    administration-transition markers.
    """
    logger.info("Creating punctuated equilibrium visualization ...")

    pivot = topics_over_time.pivot(
        index="Timestamp", columns="Topic", values="Frequency",
    ).fillna(0)
    changes = pivot.diff().abs()
    volatility = changes.sum(axis=1)

    vol_df = pd.DataFrame({"Date": volatility.index, "Volatility": volatility.values})
    vol_df.to_csv(os.path.join(output_dir, "policy_volatility.csv"), index=False)

    fig = px.line(
        vol_df, x="Date", y="Volatility",
        title="Policy Attention Volatility Over Time (Punctuated Equilibrium)",
        labels={"Volatility": "Total Topic Volatility"}, height=500,
    )

    # Administration transition markers
    for date, color, label in [
        ("2017-01-01", "red", "Trump Administration Start"),
        ("2021-01-01", "blue", "Biden Administration Start"),
    ]:
        fig.add_shape(
            type="line", x0=date, y0=0, x1=date, y1=1,
            line=dict(color=color, dash="dash"), xref="x", yref="paper",
        )
        fig.add_annotation(
            x=date, y=0.95, text=label,
            showarrow=False, xref="x", yref="paper",
        )

    fig.write_html(os.path.join(output_dir, "punctuated_equilibrium.html"))

    return vol_df

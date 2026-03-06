"""Shared utilities: logging, topic label helpers."""

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a consistently configured logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(name)


def get_topic_labels(topic_model) -> dict[int, str]:
    """Extract {topic_id: label} mapping from a fitted BERTopic model."""
    topic_info = topic_model.get_topic_info()
    return {row["Topic"]: row["Name"] for _, row in topic_info.iterrows()}

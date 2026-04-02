"""
utils.py - Data loading, preprocessing, and PySpark session helpers.

Provides dataset loading for 20 Newsgroups, text preprocessing,
and a configured PySpark session factory for local-mode execution.
"""

import logging
import re
from typing import List, Tuple

from pyspark.sql import SparkSession


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Configure root logging with a consistent format.

    Args:
        level: Logging level (default INFO).

    Returns:
        Configured logger for the calling module.
    """
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )
    return logging.getLogger(__name__)


def create_spark_session(app_name: str = "MinHashLSH") -> SparkSession:
    """
    Create a PySpark session configured for local mode on a single machine.

    Uses local[*] to utilize all available CPU cores and sets driver
    memory to 10g, leaving headroom for the OS on a 16 GB machine.

    Args:
        app_name: Name shown in the Spark UI.

    Returns:
        Active SparkSession.
    """
    spark = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "10g")
        .config("spark.executor.memory", "10g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.default.parallelism", "8")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_newsgroups(categories: List[str] = None) -> List[Tuple[int, str]]:
    """
    Load the 20 Newsgroups dataset via scikit-learn.

    Strips headers, footers, and quoted replies to reduce noise,
    as recommended when using this corpus for deduplication tasks.

    Args:
        categories: Optional list of newsgroup category names to load.
                    If None, loads all 20 categories.

    Returns:
        List of (doc_id, text) tuples, one per document.
    """
    from sklearn.datasets import fetch_20newsgroups

    logger = logging.getLogger(__name__)
    logger.info("Loading 20 Newsgroups dataset...")

    dataset = fetch_20newsgroups(
        subset="all",
        categories=categories,
        remove=("headers", "footers", "quotes"),
        random_state=42,
    )

    docs = [(i, doc) for i, doc in enumerate(dataset.data)]
    logger.info("Loaded %d documents.", len(docs))
    return docs


def preprocess_text(text: str) -> str:
    """
    Normalize a document string for shingling.

    Converts to lowercase and strips all characters except
    alphanumeric and spaces. Collapses repeated whitespace.

    Args:
        text: Raw document string.

    Returns:
        Cleaned, lowercased text string.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

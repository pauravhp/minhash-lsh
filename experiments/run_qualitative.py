"""
run_qualitative.py - Experiment 4 (Stretch): Qualitative Duplicate Examples.

Runs the full pipeline on the 20 Newsgroups dataset and shows the top
near-duplicate pairs with document excerpts.

Inspired by Lee et al. (2022), who presented qualitative examples of
near-duplicate training examples found in C4 and The Pile.

Usage:
    python experiments/run_qualitative.py

Output:
    results/qualitative_examples.txt
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import run_pipeline
from src.utils import create_spark_session, load_newsgroups, preprocess_text

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    "k": 5,
    "num_hashes": 128,
    "num_bands": 16,
    "rows_per_band": 8,
    "similarity_threshold": 0.5,
    "top_n": 20,              # Number of top pairs to show
    "excerpt_chars": 300,     # Characters to include per document excerpt
    "output_file": "results/qualitative_examples.txt",
}
# ============================================================


def format_pair(rank, doc_id_a, doc_id_b, jaccard, text_a, text_b, excerpt_chars):
    """
    Format a near-duplicate pair as a readable text block.

    Args:
        rank: Rank of this pair by similarity (1 = most similar).
        doc_id_a, doc_id_b: Document IDs.
        jaccard: True Jaccard similarity score.
        text_a, text_b: Raw document texts.
        excerpt_chars: Number of characters to include per excerpt.

    Returns:
        Formatted string block.
    """
    sep = "=" * 70
    inner_sep = "-" * 70
    excerpt_a = text_a[:excerpt_chars].replace("\n", " ").strip()
    excerpt_b = text_b[:excerpt_chars].replace("\n", " ").strip()
    lines = [
        sep,
        f"Rank #{rank} | Doc {doc_id_a} vs Doc {doc_id_b} | Jaccard = {jaccard:.4f}",
        inner_sep,
        f"[Doc {doc_id_a}]",
        excerpt_a + ("..." if len(text_a) > excerpt_chars else ""),
        inner_sep,
        f"[Doc {doc_id_b}]",
        excerpt_b + ("..." if len(text_b) > excerpt_chars else ""),
        "",
    ]
    return "\n".join(lines)


def main():
    """Run Experiment 4: qualitative near-duplicate examples."""
    os.makedirs("results", exist_ok=True)

    logger.info("=" * 60)
    logger.info("Experiment 4 (Stretch): Qualitative Examples")
    logger.info("=" * 60)

    spark = create_spark_session("Qualitative")
    sc = spark.sparkContext

    # Load all documents and keep a raw text lookup.
    logger.info("Loading 20 Newsgroups dataset...")
    all_docs = load_newsgroups()
    raw_text = {doc_id: text for doc_id, text in all_docs}

    processed = [(doc_id, preprocess_text(text)) for doc_id, text in all_docs]
    docs_rdd = sc.parallelize(processed, numSlices=8)

    # Run pipeline.
    logger.info("Running full MinHash-LSH pipeline...")
    result = run_pipeline(
        docs_rdd=docs_rdd,
        k=CONFIG["k"],
        num_hashes=CONFIG["num_hashes"],
        num_bands=CONFIG["num_bands"],
        rows_per_band=CONFIG["rows_per_band"],
        threshold=CONFIG["similarity_threshold"],
    )

    confirmed = result["confirmed_pairs"]
    logger.info("Found %d confirmed near-duplicate pairs.", len(confirmed))

    # Sort by Jaccard descending to get top pairs.
    confirmed.sort(key=lambda triple: triple[2], reverse=True)
    top_pairs = confirmed[:CONFIG["top_n"]]

    # Write output file.
    output_path = CONFIG["output_file"]
    with open(output_path, "w", encoding="utf-8") as f:
        header = [
            "Near-Duplicate Document Pairs",
            f"Dataset: 20 Newsgroups ({len(all_docs)} documents)",
            f"Pipeline: n={CONFIG['num_hashes']}, b={CONFIG['num_bands']}, "
            f"r={CONFIG['rows_per_band']}, k={CONFIG['k']}, "
            f"threshold={CONFIG['similarity_threshold']}",
            f"Total confirmed pairs: {len(confirmed)}",
            f"Showing top {len(top_pairs)} pairs by Jaccard similarity",
            "=" * 70,
            "",
        ]
        f.write("\n".join(header) + "\n")

        for rank, (doc_a, doc_b, jaccard) in enumerate(top_pairs, start=1):
            text_a = raw_text.get(doc_a, "[text not found]")
            text_b = raw_text.get(doc_b, "[text not found]")
            block = format_pair(
                rank, doc_a, doc_b, jaccard, text_a, text_b,
                CONFIG["excerpt_chars"]
            )
            f.write(block + "\n")

    logger.info("Qualitative examples written to %s", output_path)
    logger.info("=" * 60)
    logger.info("Experiment 4 complete.")

    spark.stop()


if __name__ == "__main__":
    main()

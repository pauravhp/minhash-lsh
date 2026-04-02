"""
run_scalability.py - Experiment 2: Pipeline Scalability vs. Dataset Size.

Measures wall-clock runtime of the MinHash-LSH pipeline and brute-force
baseline across increasing dataset fractions.

Usage:
    python experiments/run_scalability.py

Configuration:
    Modify the CONFIG section below to adjust parameters.

Output:
    results/scalability.csv
"""

import csv
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lsh import lsh_candidates
from src.minhash import compute_signatures_rdd
from src.pipeline import run_brute_force
from src.shingling import shingle_document_rdd
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
    "k": 5,                       # Character shingle size
    "num_hashes": 128,            # MinHash signature length (n)
    "num_bands": 16,              # Best (b) from Experiment 1
    "rows_per_band": 8,           # Best (r) from Experiment 1
    "similarity_threshold": 0.5,  # Jaccard threshold
    "dataset_fractions": [0.10, 0.25, 0.50, 1.00],
    "brute_force_max_docs": 5000, # Skip brute force above this size
    "output_file": "results/scalability.csv",
}
# ============================================================


def run_lsh_pipeline_timed(sc, processed_docs, k, num_hashes, num_bands, rows_per_band):
    """
    Run the LSH pipeline on a list of (doc_id, text) pairs and measure runtime.

    Returns (num_pairs_found, runtime_seconds).

    Args:
        sc: SparkContext.
        processed_docs: List of (doc_id, preprocessed_text) tuples.
        k, num_hashes, num_bands, rows_per_band: Pipeline parameters.

    Returns:
        Tuple of (num_pairs, runtime_seconds).
    """
    t0 = time.time()
    docs_rdd = sc.parallelize(processed_docs, numSlices=8)
    shingle_rdd = shingle_document_rdd(docs_rdd, k).cache()
    sig_rdd = compute_signatures_rdd(shingle_rdd, num_hashes).cache()
    candidate_rdd = lsh_candidates(sig_rdd, num_bands, rows_per_band)
    num_pairs = candidate_rdd.count()
    runtime = time.time() - t0
    shingle_rdd.unpersist()
    sig_rdd.unpersist()
    return num_pairs, runtime


def run_brute_force_timed(sc, processed_docs, k, threshold):
    """
    Run brute-force all-pairs Jaccard and measure runtime.

    Returns (num_pairs_found, runtime_seconds).

    Args:
        sc: SparkContext.
        processed_docs: List of (doc_id, preprocessed_text) tuples.
        k: Shingle size.
        threshold: Minimum Jaccard similarity.

    Returns:
        Tuple of (num_pairs, runtime_seconds).
    """
    t0 = time.time()
    docs_rdd = sc.parallelize(processed_docs, numSlices=8)
    shingle_rdd = shingle_document_rdd(docs_rdd, k)
    pairs = run_brute_force(shingle_rdd, threshold)
    runtime = time.time() - t0
    return len(pairs), runtime


def main():
    """Run Experiment 2: scalability analysis."""
    os.makedirs("results", exist_ok=True)

    logger.info("=" * 60)
    logger.info("Experiment 2: Pipeline Scalability")
    logger.info("=" * 60)
    logger.info(
        "Config: n=%d, b=%d, r=%d, k=%d, threshold=%.2f",
        CONFIG["num_hashes"], CONFIG["num_bands"], CONFIG["rows_per_band"],
        CONFIG["k"], CONFIG["similarity_threshold"],
    )

    spark = create_spark_session("Scalability")
    sc = spark.sparkContext

    # Load and preprocess all documents once.
    logger.info("Loading and preprocessing 20 Newsgroups...")
    all_docs = load_newsgroups()
    all_processed = [(doc_id, preprocess_text(text)) for doc_id, text in all_docs]
    total = len(all_processed)
    logger.info("Total documents available: %d", total)

    rows_out = []

    for fraction in CONFIG["dataset_fractions"]:
        n_docs = max(1, int(total * fraction))
        subset = all_processed[:n_docs]

        logger.info("-" * 40)
        logger.info("Fraction=%.2f | n_docs=%d", fraction, n_docs)

        # LSH pipeline.
        logger.info("  Running LSH pipeline...")
        lsh_pairs, lsh_time = run_lsh_pipeline_timed(
            sc, subset,
            CONFIG["k"], CONFIG["num_hashes"],
            CONFIG["num_bands"], CONFIG["rows_per_band"],
        )
        logger.info("  LSH: %d candidates in %.1fs", lsh_pairs, lsh_time)
        rows_out.append({
            "num_docs": n_docs,
            "fraction": fraction,
            "method": "lsh",
            "runtime_seconds": round(lsh_time, 2),
            "num_pairs_found": lsh_pairs,
        })

        # Brute-force baseline (only for small subsets).
        if n_docs <= CONFIG["brute_force_max_docs"]:
            logger.info("  Running brute-force baseline (n_docs=%d <= %d)...",
                        n_docs, CONFIG["brute_force_max_docs"])
            bf_pairs, bf_time = run_brute_force_timed(
                sc, subset, CONFIG["k"], CONFIG["similarity_threshold"]
            )
            logger.info("  Brute-force: %d pairs in %.1fs", bf_pairs, bf_time)
            rows_out.append({
                "num_docs": n_docs,
                "fraction": fraction,
                "method": "brute_force",
                "runtime_seconds": round(bf_time, 2),
                "num_pairs_found": bf_pairs,
            })
        else:
            logger.info(
                "  Skipping brute-force: n_docs=%d > %d limit.",
                n_docs, CONFIG["brute_force_max_docs"]
            )

    # Write CSV.
    output_path = CONFIG["output_file"]
    fieldnames = ["num_docs", "fraction", "method", "runtime_seconds", "num_pairs_found"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    logger.info("Results written to %s", output_path)
    logger.info("=" * 60)
    logger.info("Experiment 2 complete.")

    spark.stop()


if __name__ == "__main__":
    main()

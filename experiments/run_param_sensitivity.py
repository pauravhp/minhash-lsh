"""
run_param_sensitivity.py - Experiment 1: LSH Parameter Sensitivity.

Measures precision, recall, and F1 of candidate detection for varying
(b, r) LSH configurations against brute-force ground truth.

Usage:
    python experiments/run_param_sensitivity.py

Configuration:
    Modify the CONFIG section below to adjust parameters.

Output:
    results/param_sensitivity.csv
"""

import csv
import logging
import os
import sys
import time

# Add project root to path.
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
# CONFIGURATION - edit these values to change experiment params
# ============================================================
CONFIG = {
    "k": 5,                   # Character shingle size
    "num_hashes": 128,        # MinHash signature length (n); must equal b*r
    "subset_size": 2000,      # Number of docs for brute-force ground truth
    "similarity_threshold": 0.5,  # Jaccard threshold for near-duplicate
    # (b, r) configurations where b * r = 128
    "band_configs": [
        (4, 32),
        (8, 16),
        (16, 8),
        (32, 4),
        (64, 2),
        (128, 1),
    ],
    "output_file": "results/param_sensitivity.csv",
}
# ============================================================


def compute_ground_truth(shingle_rdd, threshold: float) -> set:
    """
    Compute the ground-truth near-duplicate pairs via brute force.

    Returns the set of (doc_id_a, doc_id_b) pairs with Jaccard >= threshold,
    with doc_id_a < doc_id_b.

    Args:
        shingle_rdd: RDD of (doc_id, shingle_set) for the subset.
        threshold: Minimum Jaccard similarity.

    Returns:
        Set of (doc_id_a, doc_id_b) tuples.
    """
    logger.info("Computing brute-force ground truth (threshold=%.2f)...", threshold)
    t0 = time.time()
    pairs = run_brute_force(shingle_rdd, threshold)
    elapsed = time.time() - t0
    gt_set = {(a, b) for a, b, _ in pairs}
    logger.info(
        "Ground truth: %d pairs found in %.1fs.", len(gt_set), elapsed
    )
    return gt_set


def evaluate(candidates: set, ground_truth: set) -> dict:
    """
    Compute precision, recall, and F1 given candidate and ground-truth sets.

    Args:
        candidates: Set of (doc_id_a, doc_id_b) candidate pairs from LSH.
        ground_truth: Set of (doc_id_a, doc_id_b) true near-duplicate pairs.

    Returns:
        Dict with precision, recall, f1.
    """
    if not ground_truth and not candidates:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    tp = len(candidates & ground_truth)
    precision = tp / len(candidates) if candidates else 0.0
    recall = tp / len(ground_truth) if ground_truth else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def main():
    """Run Experiment 1: LSH parameter sensitivity analysis."""
    os.makedirs("results", exist_ok=True)

    logger.info("=" * 60)
    logger.info("Experiment 1: LSH Parameter Sensitivity")
    logger.info("=" * 60)
    logger.info("Config: n=%d, k=%d, subset=%d docs, threshold=%.2f",
                CONFIG["num_hashes"], CONFIG["k"],
                CONFIG["subset_size"], CONFIG["similarity_threshold"])

    spark = create_spark_session("ParamSensitivity")
    sc = spark.sparkContext

    # Load and preprocess dataset.
    logger.info("Loading 20 Newsgroups dataset...")
    all_docs = load_newsgroups()
    subset = all_docs[:CONFIG["subset_size"]]
    processed = [(doc_id, preprocess_text(text)) for doc_id, text in subset]

    # Shingle all docs.
    logger.info("Shingling %d documents (k=%d)...", len(processed), CONFIG["k"])
    docs_rdd = sc.parallelize(processed, numSlices=8)
    shingle_rdd = shingle_document_rdd(docs_rdd, CONFIG["k"]).cache()
    shingle_rdd.count()
    logger.info("Shingling complete.")

    # Compute MinHash signatures (shared across all (b,r) configs).
    logger.info("Computing MinHash signatures (n=%d)...", CONFIG["num_hashes"])
    sig_rdd = compute_signatures_rdd(shingle_rdd, CONFIG["num_hashes"]).cache()
    sig_rdd.count()
    logger.info("Signatures computed.")

    # Ground truth via brute force.
    ground_truth = compute_ground_truth(shingle_rdd, CONFIG["similarity_threshold"])

    logger.info("Starting parameter sweep over %d (b,r) configs...",
                len(CONFIG["band_configs"]))

    rows_out = []
    for b, r in CONFIG["band_configs"]:
        assert b * r == CONFIG["num_hashes"], (
            f"Invalid config: b={b}, r={r}, b*r={b*r} != n={CONFIG['num_hashes']}"
        )
        logger.info("Testing (b=%d, r=%d)...", b, r)
        t0 = time.time()

        # LSH candidate generation.
        raw_candidates = lsh_candidates(sig_rdd, num_bands=b, rows_per_band=r)
        candidate_set = set(raw_candidates.collect())
        num_candidates = len(candidate_set)

        # Evaluation vs ground truth.
        metrics = evaluate(candidate_set, ground_truth)
        runtime = time.time() - t0

        logger.info(
            "  b=%d r=%d: %d candidates | P=%.3f R=%.3f F1=%.3f | %.1fs",
            b, r, num_candidates,
            metrics["precision"], metrics["recall"], metrics["f1"],
            runtime,
        )

        rows_out.append({
            "b": b,
            "r": r,
            "threshold": CONFIG["similarity_threshold"],
            "num_candidates": num_candidates,
            "num_ground_truth": len(ground_truth),
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1": round(metrics["f1"], 4),
            "runtime_seconds": round(runtime, 2),
        })

    # Write CSV.
    output_path = CONFIG["output_file"]
    fieldnames = ["b", "r", "threshold", "num_candidates", "num_ground_truth",
                  "precision", "recall", "f1", "runtime_seconds"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    logger.info("Results written to %s", output_path)
    logger.info("=" * 60)
    logger.info("Experiment 1 complete.")

    shingle_rdd.unpersist()
    sig_rdd.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()

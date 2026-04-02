"""
run_signature_length.py - Experiment 3 (Stretch): MinHash Signature Length vs. Error.

Measures how MinHash estimation error decreases as the signature length n increases.
Compares empirical mean absolute error to the theoretical bound 1/sqrt(n).

Broder (1997): the standard deviation of the MinHash estimator is approximately
sqrt(J(1-J)/n), which is bounded above by 1/(2*sqrt(n)).

Usage:
    python experiments/run_signature_length.py

Output:
    results/signature_length.csv
"""

import csv
import logging
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.minhash import compute_minhash_signature, generate_hash_functions
from src.shingling import generate_shingles
from src.utils import load_newsgroups, preprocess_text
from src.verification import true_jaccard

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
    "k": 5,                    # Character shingle size
    "num_doc_sample": 5000,    # Number of documents to sample
    "num_pair_sample": 10000,  # Number of random pairs to evaluate
    "n_values": [32, 64, 128, 256, 512],  # Signature lengths to test
    "random_seed": 42,
    "output_file": "results/signature_length.csv",
}
# ============================================================


def main():
    """Run Experiment 3: signature length vs. Jaccard estimation error."""
    os.makedirs("results", exist_ok=True)

    logger.info("=" * 60)
    logger.info("Experiment 3 (Stretch): Signature Length vs. Error")
    logger.info("=" * 60)

    rng = random.Random(CONFIG["random_seed"])

    # Load and preprocess documents.
    logger.info("Loading dataset...")
    all_docs = load_newsgroups()
    sample = all_docs[:CONFIG["num_doc_sample"]]
    logger.info("Preprocessing %d documents...", len(sample))
    processed = [(doc_id, preprocess_text(text)) for doc_id, text in sample]

    # Shingle all documents once (shared across all n values).
    logger.info("Shingling all documents (k=%d)...", CONFIG["k"])
    shingle_dict = {
        doc_id: generate_shingles(text, CONFIG["k"])
        for doc_id, text in processed
    }

    doc_ids = list(shingle_dict.keys())
    n_docs = len(doc_ids)

    # Sample random pairs.
    logger.info("Sampling %d random pairs...", CONFIG["num_pair_sample"])
    pairs = []
    while len(pairs) < CONFIG["num_pair_sample"]:
        i = rng.randint(0, n_docs - 1)
        j = rng.randint(0, n_docs - 1)
        if i != j:
            pairs.append((doc_ids[i], doc_ids[j]))

    # Compute true Jaccard for all sampled pairs.
    logger.info("Computing true Jaccard for all pairs...")
    true_jaccards = [
        true_jaccard(shingle_dict[a], shingle_dict[b])
        for a, b in pairs
    ]

    rows_out = []

    for n in CONFIG["n_values"]:
        logger.info("Testing n=%d...", n)
        t0 = time.time()

        params = generate_hash_functions(n, seed=CONFIG["random_seed"])

        # Pre-compute signatures for each document in the pairs.
        seen = set(a for a, b in pairs) | set(b for a, b in pairs)
        sigs = {
            doc_id: compute_minhash_signature(shingle_dict[doc_id], params)
            for doc_id in seen
        }

        # Compute estimated Jaccard from signatures and error.
        errors = []
        for (a, b), true_j in zip(pairs, true_jaccards):
            sig_a = sigs[a]
            sig_b = sigs[b]
            matches = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
            est_j = matches / n
            errors.append(abs(est_j - true_j))

        mean_err = sum(errors) / len(errors)
        std_err = math.sqrt(sum((e - mean_err) ** 2 for e in errors) / len(errors))
        theoretical_bound = 1.0 / math.sqrt(n)
        runtime = time.time() - t0

        logger.info(
            "  n=%d: MAE=%.4f, std=%.4f, theory=%.4f | %.1fs",
            n, mean_err, std_err, theoretical_bound, runtime
        )

        rows_out.append({
            "n": n,
            "mean_absolute_error": round(mean_err, 6),
            "std_error": round(std_err, 6),
            "theoretical_bound_1_over_sqrt_n": round(theoretical_bound, 6),
            "num_pairs": len(pairs),
            "runtime_seconds": round(runtime, 2),
        })

    output_path = CONFIG["output_file"]
    fieldnames = ["n", "mean_absolute_error", "std_error",
                  "theoretical_bound_1_over_sqrt_n", "num_pairs", "runtime_seconds"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    logger.info("Results written to %s", output_path)
    logger.info("=" * 60)
    logger.info("Experiment 3 complete.")


if __name__ == "__main__":
    main()

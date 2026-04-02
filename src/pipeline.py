"""
pipeline.py - End-to-end MinHash-LSH pipeline orchestration.

Ties together the three phases of near-duplicate detection:
    Phase 1: Shingling + MinHash signature generation (embarrassingly parallel)
    Phase 2: LSH banding for candidate pair generation (MapReduce)
    Phase 3: Candidate verification via true Jaccard similarity

Reference:
    Broder, A. Z. (1997). "On the Resemblance and Containment of Documents."
    Lee et al. (2022). "Deduplicating Training Data Makes Language Models Better." ACL.
"""

import logging
import time
from itertools import combinations
from typing import Dict, List, Tuple

from pyspark import RDD

from src.lsh import lsh_candidates
from src.minhash import compute_signatures_rdd
from src.shingling import shingle_document_rdd
from src.verification import true_jaccard, verify_candidates

logger = logging.getLogger(__name__)


def run_pipeline(
    docs_rdd: RDD,
    k: int,
    num_hashes: int,
    num_bands: int,
    rows_per_band: int,
    threshold: float,
) -> Dict:
    """
    Run the complete MinHash-LSH near-duplicate detection pipeline.

    Executes three phases sequentially, timing each, and returns
    a summary dictionary with confirmed pairs and per-phase runtimes.

    Pipeline phases (Broder 1997 + MMDS Chapter 3):
        1. Shingling: (doc_id, text) -> (doc_id, Set[int])
        2. MinHash:   (doc_id, Set[int]) -> (doc_id, List[int])
        3. LSH:       (doc_id, List[int]) -> (doc_id_a, doc_id_b) candidates
        4. Verify:    candidates -> (doc_id_a, doc_id_b, jaccard) confirmed pairs

    Args:
        docs_rdd: RDD of (doc_id: int, text: str) tuples.
        k: Character shingle size.
        num_hashes: MinHash signature length (n).
        num_bands: Number of LSH bands (b).
        rows_per_band: Rows per LSH band (r). Must satisfy b * r == num_hashes.
        threshold: Minimum true Jaccard similarity for confirmed pairs.

    Returns:
        Dict with keys:
            confirmed_pairs: List of (doc_id_a, doc_id_b, jaccard) tuples
            num_candidates: Count of candidate pairs before verification
            num_confirmed: Count of pairs with jaccard >= threshold
            runtime_shingling: Wall-clock seconds for Phase 1
            runtime_minhash: Wall-clock seconds for Phase 2
            runtime_lsh: Wall-clock seconds for Phase 3
            runtime_verification: Wall-clock seconds for Phase 4
            runtime_total: Total wall-clock seconds
    """
    assert num_bands * rows_per_band == num_hashes, (
        f"num_bands ({num_bands}) * rows_per_band ({rows_per_band}) "
        f"must equal num_hashes ({num_hashes})."
    )

    total_start = time.time()

    # Phase 1: Shingling
    logger.info("Phase 1: Shingling (k=%d)...", k)
    t0 = time.time()
    shingle_rdd = shingle_document_rdd(docs_rdd, k).cache()
    # Force evaluation by counting.
    num_docs = shingle_rdd.count()
    runtime_shingling = time.time() - t0
    logger.info("Phase 1 done: %d documents shingled in %.1fs.", num_docs, runtime_shingling)

    # Phase 2: MinHash signatures
    logger.info("Phase 2: MinHash signatures (n=%d)...", num_hashes)
    t0 = time.time()
    sig_rdd = compute_signatures_rdd(shingle_rdd, num_hashes).cache()
    sig_rdd.count()  # Force evaluation.
    runtime_minhash = time.time() - t0
    logger.info("Phase 2 done in %.1fs.", runtime_minhash)

    # Phase 3: LSH candidate generation
    logger.info("Phase 3: LSH banding (b=%d, r=%d)...", num_bands, rows_per_band)
    t0 = time.time()
    candidate_rdd = lsh_candidates(sig_rdd, num_bands, rows_per_band).cache()
    num_candidates = candidate_rdd.count()
    runtime_lsh = time.time() - t0
    logger.info(
        "Phase 3 done: %d candidate pairs in %.1fs.", num_candidates, runtime_lsh
    )

    # Phase 4: Verification
    logger.info(
        "Phase 4: Verifying %d candidates (threshold=%.2f)...",
        num_candidates,
        threshold,
    )
    t0 = time.time()
    confirmed_rdd = verify_candidates(candidate_rdd, shingle_rdd, threshold)
    confirmed_pairs = confirmed_rdd.collect()
    runtime_verification = time.time() - t0
    num_confirmed = len(confirmed_pairs)
    logger.info(
        "Phase 4 done: %d confirmed pairs in %.1fs.", num_confirmed, runtime_verification
    )

    runtime_total = time.time() - total_start
    logger.info("Pipeline complete. Total time: %.1fs.", runtime_total)

    # Unpersist cached RDDs.
    shingle_rdd.unpersist()
    sig_rdd.unpersist()
    candidate_rdd.unpersist()

    return {
        "confirmed_pairs": confirmed_pairs,
        "num_candidates": num_candidates,
        "num_confirmed": num_confirmed,
        "runtime_shingling": runtime_shingling,
        "runtime_minhash": runtime_minhash,
        "runtime_lsh": runtime_lsh,
        "runtime_verification": runtime_verification,
        "runtime_total": runtime_total,
    }


def run_brute_force(
    shingle_rdd: RDD,
    threshold: float,
) -> List[Tuple[int, int, float]]:
    """
    Compute all-pairs Jaccard similarity via brute force.

    Brute-force baseline for ground truth comparison. O(N^2) complexity.
    Only suitable for subsets up to ~5,000 documents. Collects all shingle
    sets to the driver, then runs a nested loop over all pairs.

    Args:
        shingle_rdd: RDD of (doc_id: int, shingle_set: Set[int]) tuples.
        threshold: Minimum Jaccard similarity to include a pair.

    Returns:
        List of (doc_id_a, doc_id_b, jaccard) tuples with jaccard >= threshold.
    """
    logger.info("Brute-force all-pairs Jaccard (collecting to driver)...")
    shingle_list: List[Tuple[int, set]] = shingle_rdd.collect()
    logger.info("Running O(N^2) loop over %d documents...", len(shingle_list))

    results = []
    for (id_a, set_a), (id_b, set_b) in combinations(shingle_list, 2):
        j = true_jaccard(set_a, set_b)
        if j >= threshold:
            a, b = (id_a, id_b) if id_a < id_b else (id_b, id_a)
            results.append((a, b, j))

    logger.info("Brute-force found %d pairs with jaccard >= %.2f.", len(results), threshold)
    return results

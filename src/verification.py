"""
verification.py - True Jaccard similarity computation and candidate filtering.

For each LSH candidate pair, compute the exact Jaccard similarity from the
full shingle sets and filter out pairs below the similarity threshold.
This is the verification (post-processing) phase of the MinHash-LSH pipeline.

Reference:
    Broder, A. Z. (1997). "On the Resemblance and Containment of Documents."
    True Jaccard: J(A, B) = |A intersect B| / |A union B|
"""

import logging
from typing import Dict, Set

from pyspark import RDD

logger = logging.getLogger(__name__)


def true_jaccard(set_a: Set[int], set_b: Set[int]) -> float:
    """
    Compute exact Jaccard similarity between two shingle sets.

    J(A, B) = |A intersect B| / |A union B|

    Broder (1997): The true Jaccard similarity is the ground-truth
    resemblance measure that MinHash approximates.

    Args:
        set_a: Shingle set for document A.
        set_b: Shingle set for document B.

    Returns:
        Jaccard similarity in [0, 1]. Returns 0.0 if both sets are empty.
    """
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def verify_candidates(
    candidate_rdd: RDD,
    shingle_rdd: RDD,
    threshold: float,
) -> RDD:
    """
    Verify candidate pairs by computing true Jaccard similarity.

    Strategy: collect all shingle sets to the driver as a broadcast dict
    (feasible for up to ~18,000 documents at k=5 where each set is small),
    then map over candidate pairs to compute exact Jaccard and filter.

    Args:
        candidate_rdd: RDD of (doc_id_a: int, doc_id_b: int) candidate pairs.
        shingle_rdd: RDD of (doc_id: int, shingle_set: Set[int]) tuples.
        threshold: Minimum Jaccard similarity to keep a confirmed pair.

    Returns:
        RDD of (doc_id_a: int, doc_id_b: int, jaccard: float) tuples
        for pairs with jaccard >= threshold.
    """
    logger.info(
        "Verifying candidates with true Jaccard (threshold=%.2f).", threshold
    )

    sc = candidate_rdd.context

    # Collect shingle sets to driver and broadcast.
    shingle_dict: Dict[int, Set[int]] = dict(shingle_rdd.collect())
    bc_shingles = sc.broadcast(shingle_dict)

    def _verify(pair):
        doc_a, doc_b = pair
        shingles = bc_shingles.value
        set_a = shingles.get(doc_a, set())
        set_b = shingles.get(doc_b, set())
        jaccard = true_jaccard(set_a, set_b)
        return (doc_a, doc_b, jaccard)

    return (
        candidate_rdd
        .map(_verify)
        .filter(lambda triple: triple[2] >= threshold)
    )

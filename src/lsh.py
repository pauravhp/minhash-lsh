"""
lsh.py - Locality-Sensitive Hashing via banding for candidate pair generation.

Implements the banding technique from MMDS Chapter 3.4. The signature matrix
is divided into b bands of r rows each. Two documents are candidate duplicates
if they hash to the same bucket in at least one band.

Reference:
    Leskovec, Rajaraman, Ullman. "Mining of Massive Datasets," Chapter 3.4.
    Broder, A. Z. (1997). "On the Resemblance and Containment of Documents."
"""

import hashlib
import logging
import struct
from itertools import combinations
from typing import List, Tuple

from pyspark import RDD

logger = logging.getLogger(__name__)


def lsh_candidates(
    signatures_rdd: RDD,
    num_bands: int,
    rows_per_band: int,
) -> RDD:
    """
    Generate candidate near-duplicate pairs using the LSH banding technique.

    Pipeline:
        1. flatMap: for each doc, emit ((band_idx, band_hash), doc_id) per band.
        2. groupByKey: collect all doc_ids that collide in each bucket.
        3. Filter buckets with 2+ docs.
        4. Emit all (doc_id_a, doc_id_b) pairs from each bucket.
        5. Deduplicate: normalize so doc_id_a < doc_id_b, then distinct().

    MMDS Chapter 3.4: A pair (A, B) becomes a candidate if their signatures
    agree in all r rows of at least one band. The probability of this is:
        P(candidate) = 1 - (1 - s^r)^b
    where s = J(A, B).

    Args:
        signatures_rdd: RDD of (doc_id: int, signature: List[int]) tuples.
        num_bands: Number of bands (b).
        rows_per_band: Number of signature rows per band (r).

    Returns:
        RDD of (doc_id_a: int, doc_id_b: int) candidate pairs,
        where doc_id_a < doc_id_b, deduplicated.
    """
    logger.info(
        "Running LSH banding: b=%d bands, r=%d rows per band.", num_bands, rows_per_band
    )

    def emit_band_buckets(pair):
        """For one document, emit one (bucket_key, doc_id) per band.

        Uses a deterministic MD5-based hash of the band slice so that
        identical slices produce the same bucket key across all Spark
        worker processes regardless of PYTHONHASHSEED.
        """
        doc_id, signature = pair
        items = []
        for band_idx in range(num_bands):
            start = band_idx * rows_per_band
            end = start + rows_per_band
            band_slice = signature[start:end]
            # Pack slice values as 8-byte signed longs for deterministic hashing.
            packed = struct.pack(f"{len(band_slice)}q", *band_slice)
            digest = hashlib.md5(packed, usedforsecurity=False).digest()
            band_hash = int.from_bytes(digest[:8], "little")
            bucket_key = (band_idx, band_hash)
            items.append((bucket_key, doc_id))
        return items

    def emit_candidate_pairs(bucket_and_docs):
        """From a bucket containing multiple docs, emit all distinct pairs."""
        _bucket_key, doc_ids = bucket_and_docs
        doc_list = list(doc_ids)
        pairs = []
        for a, b in combinations(doc_list, 2):
            # Normalize so the smaller id is always first.
            if a > b:
                a, b = b, a
            pairs.append((a, b))
        return pairs

    candidate_rdd = (
        signatures_rdd
        .flatMap(emit_band_buckets)
        .groupByKey()
        .filter(lambda kv: len(list(kv[1])) >= 2)
        # Re-materialize the groupByKey iterator before passing to emit_candidate_pairs.
        .map(lambda kv: (kv[0], list(kv[1])))
        .flatMap(emit_candidate_pairs)
        .distinct()
    )

    return candidate_rdd


def compute_s_curve(similarity: float, bands: int, rows: int) -> float:
    """
    Compute the theoretical LSH S-curve probability for a given similarity.

    Returns the probability that a pair with Jaccard similarity `similarity`
    is detected as a candidate pair under the (b, r) banding scheme.

    MMDS Chapter 3.4: P(candidate | s) = 1 - (1 - s^r)^b

    Args:
        similarity: True Jaccard similarity in [0, 1].
        bands: Number of bands (b).
        rows: Number of rows per band (r).

    Returns:
        Probability of the pair being a candidate in [0, 1].
    """
    return 1.0 - (1.0 - similarity ** rows) ** bands

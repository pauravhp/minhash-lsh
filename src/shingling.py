"""
shingling.py - Character k-shingle generation and hashing.

Implements document-to-set conversion via character k-shingling,
as described in Broder (1997) Section 2 on resemblance via shingling.

Reference:
    Broder, A. Z. (1997). "On the Resemblance and Containment of Documents."
    IEEE SEQUENCES 1997.
"""

import hashlib
import logging
from typing import Set

from pyspark import RDD

logger = logging.getLogger(__name__)


def _hash_shingle(shingle: str) -> int:
    """
    Hash a shingle string to a non-negative 32-bit integer deterministically.

    Uses MD5 so that the result is identical across Python processes regardless
    of PYTHONHASHSEED, which is essential for correctness in PySpark where
    different worker processes have different hash seeds.

    Args:
        shingle: k-character substring to hash.

    Returns:
        Non-negative 32-bit integer hash value.
    """
    digest = hashlib.md5(shingle.encode("utf-8"), usedforsecurity=False).digest()
    # Take first 4 bytes as a little-endian unsigned int.
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def generate_shingles(text: str, k: int) -> Set[int]:
    """
    Extract all character k-shingles from a document and hash each to an integer.

    A k-shingle is a contiguous substring of k characters. Each shingle is
    hashed to a 32-bit integer using MD5 for cross-process determinism.

    Broder (1997) Section 2: documents are represented as sets of k-shingles
    so that Jaccard similarity of the sets approximates textual resemblance.

    Args:
        text: Preprocessed document string.
        k: Shingle size (number of characters per shingle).

    Returns:
        Set of integer-hashed k-shingles.
    """
    if len(text) < k:
        # Document shorter than shingle size: treat the whole text as one shingle.
        return {_hash_shingle(text)}

    shingles: Set[int] = set()
    for i in range(len(text) - k + 1):
        shingles.add(_hash_shingle(text[i : i + k]))
    return shingles


def shingle_document_rdd(rdd: RDD, k: int) -> RDD:
    """
    Apply shingling to all documents in a PySpark RDD in parallel.

    This is a pure map transformation (no shuffling), making it
    embarrassingly parallel across all partitions.

    Args:
        rdd: RDD of (doc_id: int, text: str) tuples.
        k: Shingle size in characters.

    Returns:
        RDD of (doc_id: int, shingle_set: Set[int]) tuples.
    """
    logger.info("Shingling documents with k=%d", k)
    return rdd.map(lambda pair: (pair[0], generate_shingles(pair[1], k)))

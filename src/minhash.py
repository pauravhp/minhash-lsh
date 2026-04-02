"""
minhash.py - MinHash signature computation via universal hashing.

Implements the MinHash algorithm using a universal hash family
h_i(x) = (a_i * x + b_i) mod p to approximate min-wise independent
permutations, as described in Broder (1997).

Reference:
    Broder, A. Z. (1997). "On the Resemblance and Containment of Documents."
    IEEE SEQUENCES 1997.
    Key theorem: P[min(h(A)) = min(h(B))] = J(A, B)
"""

import logging
import random
from typing import List, Set, Tuple

from pyspark import Broadcast, RDD

logger = logging.getLogger(__name__)

# Large Mersenne prime used as modulus for universal hash family.
# Must be larger than the maximum possible shingle hash value (2^31 - 1).
LARGE_PRIME = (1 << 61) - 1  # Mersenne prime 2^61 - 1


def generate_hash_functions(
    num_hashes: int,
    seed: int = 42,
) -> List[Tuple[int, int]]:
    """
    Generate random (a, b) coefficients for a universal hash family.

    Each hash function is h_i(x) = (a_i * x + b_i) mod p, where p is
    LARGE_PRIME. The (a, b) pairs are sampled uniformly from [1, p-1]
    and [0, p-1] respectively.

    Broder (1997): min-wise independent permutations can be approximated
    via a family of universal hash functions applied to shingle IDs.

    Args:
        num_hashes: Number of hash functions (equals signature length n).
        seed: Random seed for reproducibility.

    Returns:
        List of (a_i, b_i) coefficient tuples, one per hash function.
    """
    rng = random.Random(seed)
    params: List[Tuple[int, int]] = []
    for _ in range(num_hashes):
        a = rng.randint(1, LARGE_PRIME - 1)
        b = rng.randint(0, LARGE_PRIME - 1)
        params.append((a, b))
    logger.debug("Generated %d hash functions.", num_hashes)
    return params


def compute_minhash_signature(
    shingle_set: Set[int],
    hash_params: List[Tuple[int, int]],
    prime: int = LARGE_PRIME,
) -> List[int]:
    """
    Compute the MinHash signature vector for a single document's shingle set.

    For each hash function h_i, the signature entry is min_{x in S} h_i(x).
    The resulting vector has length equal to num_hashes (n).

    Broder (1997): P[min(h(A)) = min(h(B))] = J(A, B).
    Therefore the fraction of signature positions where two docs agree
    is an unbiased estimator of their Jaccard similarity.

    Args:
        shingle_set: Set of integer-hashed shingles for the document.
        hash_params: List of (a, b) coefficient tuples.
        prime: Modulus for universal hashing (default LARGE_PRIME).

    Returns:
        MinHash signature as a list of integers of length num_hashes.
    """
    if not shingle_set:
        # Empty document: return maximum integers (will never match anything).
        return [prime] * len(hash_params)

    signature: List[int] = []
    for a, b in hash_params:
        min_val = prime  # Initialize above any possible hash value mod prime.
        for x in shingle_set:
            h = (a * x + b) % prime
            if h < min_val:
                min_val = h
        signature.append(min_val)
    return signature


def compute_signatures_rdd(
    shingle_rdd: RDD,
    num_hashes: int,
    seed: int = 42,
) -> RDD:
    """
    Compute MinHash signatures for all documents in parallel via PySpark.

    Broadcasts the hash function coefficients to all workers to avoid
    repeatedly serializing them for each task.

    Args:
        shingle_rdd: RDD of (doc_id: int, shingle_set: Set[int]) tuples.
        num_hashes: Length of the MinHash signature (n).
        seed: Random seed for hash function generation.

    Returns:
        RDD of (doc_id: int, signature: List[int]) tuples.
    """
    logger.info("Computing MinHash signatures with n=%d hash functions.", num_hashes)

    sc = shingle_rdd.context
    hash_params = generate_hash_functions(num_hashes, seed=seed)
    prime = LARGE_PRIME

    # Broadcast coefficients so each worker receives them only once.
    bc_params: Broadcast = sc.broadcast(hash_params)
    bc_prime: Broadcast = sc.broadcast(prime)

    def _compute(pair):
        doc_id, shingle_set = pair
        sig = compute_minhash_signature(shingle_set, bc_params.value, bc_prime.value)
        return (doc_id, sig)

    return shingle_rdd.map(_compute)


def estimate_jaccard_from_signatures(sig_a: List[int], sig_b: List[int]) -> float:
    """
    Estimate Jaccard similarity between two documents from their signatures.

    The estimator is the fraction of signature positions where sig_a and
    sig_b agree. Broder (1997) shows this is an unbiased estimator of J(A,B).

    Args:
        sig_a: MinHash signature for document A.
        sig_b: MinHash signature for document B.

    Returns:
        Estimated Jaccard similarity in [0, 1].
    """
    if not sig_a or not sig_b:
        return 0.0
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)

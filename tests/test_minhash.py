"""
test_minhash.py - Unit tests for src/minhash.py.

Verifies that MinHash estimates converge to true Jaccard within the
expected error bound of 1/sqrt(n) for large n, as established by
Broder (1997).
"""

import math
import pytest
from src.minhash import (
    compute_minhash_signature,
    estimate_jaccard_from_signatures,
    generate_hash_functions,
    LARGE_PRIME,
)
from src.shingling import generate_shingles
from src.verification import true_jaccard


class TestGenerateHashFunctions:
    def test_length(self):
        """Must return exactly num_hashes (a, b) pairs."""
        params = generate_hash_functions(128)
        assert len(params) == 128

    def test_coefficients_are_positive_integers(self):
        """All a_i must be >= 1 and all b_i must be >= 0."""
        params = generate_hash_functions(64)
        for a, b in params:
            assert isinstance(a, int) and a >= 1
            assert isinstance(b, int) and b >= 0

    def test_reproducible_with_same_seed(self):
        """Same seed must produce identical coefficient lists."""
        p1 = generate_hash_functions(64, seed=0)
        p2 = generate_hash_functions(64, seed=0)
        assert p1 == p2

    def test_different_seeds_produce_different_params(self):
        """Different seeds should (almost certainly) differ."""
        p1 = generate_hash_functions(64, seed=1)
        p2 = generate_hash_functions(64, seed=2)
        assert p1 != p2


class TestComputeMinHashSignature:
    def test_signature_length(self):
        """Signature length must equal number of hash functions."""
        shingle_set = {1, 2, 3, 4, 5}
        params = generate_hash_functions(100)
        sig = compute_minhash_signature(shingle_set, params)
        assert len(sig) == 100

    def test_empty_set(self):
        """Empty shingle set must return a signature of LARGE_PRIME values."""
        params = generate_hash_functions(32)
        sig = compute_minhash_signature(set(), params)
        assert all(v == LARGE_PRIME for v in sig)

    def test_identical_sets_same_signature(self):
        """Identical shingle sets must produce identical signatures."""
        params = generate_hash_functions(64, seed=7)
        s = {10, 20, 30, 40}
        assert compute_minhash_signature(s, params) == compute_minhash_signature(s, params)

    def test_signature_values_are_nonneg(self):
        """All signature values must be non-negative."""
        params = generate_hash_functions(50)
        sig = compute_minhash_signature({100, 200, 300}, params)
        assert all(v >= 0 for v in sig)


class TestMinHashAccuracy:
    """
    Broder (1997): the expected estimation error for n hashes is 1/sqrt(n).
    With n=200, error bound is ~0.071. We use a tolerance of 0.15 to be robust.
    """

    def _make_pair_with_jaccard(self, jaccard_target: float, universe_size: int = 1000):
        """
        Construct two sets A, B where |A intersect B| / |A union B| ~ jaccard_target.
        """
        # Let |A union B| = universe_size, |A intersect B| = round(jaccard_target * universe_size).
        intersection_size = round(jaccard_target * universe_size)
        union_size = universe_size
        A = set(range(intersection_size))
        only_a_size = (union_size - intersection_size) // 2
        only_b_size = union_size - intersection_size - only_a_size
        B = set(range(intersection_size))
        A |= set(range(intersection_size, intersection_size + only_a_size))
        B |= set(range(intersection_size + only_a_size, intersection_size + only_a_size + only_b_size))
        return A, B

    @pytest.mark.parametrize("true_j", [0.2, 0.5, 0.8])
    def test_estimate_within_error_bound(self, true_j: float):
        """MinHash estimate must be within 0.15 of true Jaccard for n=200."""
        n = 200
        A, B = self._make_pair_with_jaccard(true_j, universe_size=2000)
        actual_j = true_jaccard(A, B)

        params = generate_hash_functions(n, seed=42)
        sig_a = compute_minhash_signature(A, params)
        sig_b = compute_minhash_signature(B, params)
        estimated_j = estimate_jaccard_from_signatures(sig_a, sig_b)

        tolerance = 0.15
        assert abs(estimated_j - actual_j) < tolerance, (
            f"Jaccard={actual_j:.3f}, estimated={estimated_j:.3f}, "
            f"error={abs(estimated_j - actual_j):.3f} > {tolerance}"
        )

    def test_high_similarity_detected_as_high(self):
        """Near-identical documents (J~0.9) must be estimated as high similarity."""
        text_a = "the quick brown fox jumps over the lazy dog " * 20
        text_b = text_a + " extra sentence here"
        sa = generate_shingles(text_a, k=5)
        sb = generate_shingles(text_b, k=5)

        params = generate_hash_functions(128, seed=0)
        sig_a = compute_minhash_signature(sa, params)
        sig_b = compute_minhash_signature(sb, params)
        est = estimate_jaccard_from_signatures(sig_a, sig_b)
        assert est > 0.7, f"High-similarity pair estimated too low: {est:.3f}"

    def test_low_similarity_detected_as_low(self):
        """Completely different documents must be estimated near zero similarity."""
        sa = generate_shingles("aaaa " * 100, k=5)
        sb = generate_shingles("zzzz " * 100, k=5)

        params = generate_hash_functions(128, seed=0)
        sig_a = compute_minhash_signature(sa, params)
        sig_b = compute_minhash_signature(sb, params)
        est = estimate_jaccard_from_signatures(sig_a, sig_b)
        assert est < 0.2, f"Low-similarity pair estimated too high: {est:.3f}"

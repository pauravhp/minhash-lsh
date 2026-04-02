"""
test_lsh.py - Unit tests for src/lsh.py.

Verifies that the LSH banding algorithm correctly identifies near-duplicate
pairs and avoids false positives for dissimilar pairs.

MMDS Chapter 3.4: With b=16, r=8 (n=128), the S-curve threshold (50% detection
probability) is at s ~ (1/b)^(1/r) = (1/16)^(1/8) ~ 0.72.
"""

import pytest
from src.lsh import compute_s_curve, lsh_candidates
from src.minhash import compute_minhash_signature, generate_hash_functions
from src.shingling import generate_shingles
from src.utils import create_spark_session


@pytest.fixture(scope="module")
def spark():
    """Shared Spark session for all LSH tests."""
    session = create_spark_session("TestLSH")
    yield session
    session.stop()


def make_signature(text: str, k: int, num_hashes: int, seed: int = 42) -> list:
    """Helper: shingle + MinHash a text string."""
    shingles = generate_shingles(text, k)
    params = generate_hash_functions(num_hashes, seed=seed)
    return compute_minhash_signature(shingles, params)


class TestLSHCandidates:
    def test_similar_pair_is_candidate(self, spark):
        """
        A near-duplicate pair (J ~ 0.8) must appear as a candidate.
        Uses b=32, r=4 (n=128) which has high recall at J=0.8.
        """
        base = "the quick brown fox jumps over the lazy dog " * 30
        doc_a = base
        doc_b = base + " slightly different ending to make it a near duplicate"

        sig_a = make_signature(doc_a, k=5, num_hashes=128)
        sig_b = make_signature(doc_b, k=5, num_hashes=128)

        sig_rdd = spark.sparkContext.parallelize([
            (0, sig_a),
            (1, sig_b),
            (2, make_signature("completely unrelated xyz abc def " * 20, k=5, num_hashes=128)),
        ])

        # b=32, r=4: high recall configuration for J >= 0.7
        candidates = lsh_candidates(sig_rdd, num_bands=32, rows_per_band=4).collect()
        candidate_set = set(candidates)

        assert (0, 1) in candidate_set or (1, 0) in candidate_set, (
            f"Expected (0,1) in candidates. Got: {candidate_set}"
        )

    def test_dissimilar_pair_not_candidate(self, spark):
        """
        A dissimilar pair (J < 0.1) should not appear as a candidate
        under a strict banding config (b=8, r=16).
        This test is probabilistic but holds with overwhelming probability.
        """
        text_a = "aardvark apple avenue artifact academic " * 50
        text_b = "zebra zealot zenith zeppelin zombie " * 50

        sig_a = make_signature(text_a, k=5, num_hashes=128, seed=99)
        sig_b = make_signature(text_b, k=5, num_hashes=128, seed=99)

        sig_rdd = spark.sparkContext.parallelize([
            (0, sig_a),
            (1, sig_b),
        ])

        # b=8, r=16: very strict - almost no false positives at J < 0.1
        candidates = lsh_candidates(sig_rdd, num_bands=8, rows_per_band=16).collect()
        candidate_set = set(candidates)

        assert (0, 1) not in candidate_set and (1, 0) not in candidate_set, (
            f"Dissimilar pair incorrectly found as candidate: {candidate_set}"
        )

    def test_output_normalized_pairs(self, spark):
        """All output pairs must satisfy doc_id_a < doc_id_b (normalized)."""
        sigs = [(i, make_signature("shared text " * 20 + str(i), k=3, num_hashes=64))
                for i in range(5)]
        sig_rdd = spark.sparkContext.parallelize(sigs)

        candidates = lsh_candidates(sig_rdd, num_bands=16, rows_per_band=4).collect()
        for a, b in candidates:
            assert a < b, f"Pair not normalized: ({a}, {b})"

    def test_no_self_pairs(self, spark):
        """No document should be paired with itself."""
        sigs = [(i, make_signature("text content " * 10, k=4, num_hashes=64))
                for i in range(4)]
        sig_rdd = spark.sparkContext.parallelize(sigs)

        candidates = lsh_candidates(sig_rdd, num_bands=16, rows_per_band=4).collect()
        for a, b in candidates:
            assert a != b, f"Self-pair found: ({a}, {b})"

    def test_no_duplicate_pairs(self, spark):
        """Each candidate pair must appear at most once."""
        sigs = [(i, make_signature("shared base content " * 15, k=5, num_hashes=64))
                for i in range(6)]
        sig_rdd = spark.sparkContext.parallelize(sigs)

        candidates = lsh_candidates(sig_rdd, num_bands=16, rows_per_band=4).collect()
        assert len(candidates) == len(set(candidates)), "Duplicate pairs found."


class TestSCurve:
    def test_returns_float_in_unit_interval(self):
        """S-curve probability must be in [0, 1]."""
        for s in [0.0, 0.3, 0.5, 0.8, 1.0]:
            p = compute_s_curve(s, bands=16, rows=8)
            assert 0.0 <= p <= 1.0, f"P={p} out of range for s={s}"

    def test_monotone_increasing(self):
        """Higher similarity must give higher detection probability."""
        probs = [compute_s_curve(s / 10, bands=16, rows=8) for s in range(0, 11)]
        for i in range(len(probs) - 1):
            assert probs[i] <= probs[i + 1], "S-curve is not monotone increasing."

    def test_high_similarity_detected(self):
        """S-curve at s=0.9 with b=16,r=8 should be close to 1."""
        p = compute_s_curve(0.9, bands=16, rows=8)
        assert p > 0.95, f"Expected P > 0.95, got {p:.3f}"

    def test_low_similarity_low_probability(self):
        """S-curve at s=0.1 with b=16,r=8 should be near 0."""
        p = compute_s_curve(0.1, bands=16, rows=8)
        assert p < 0.05, f"Expected P < 0.05, got {p:.3f}"

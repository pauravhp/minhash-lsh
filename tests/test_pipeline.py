"""
test_pipeline.py - End-to-end pipeline test on a small real dataset.

Loads 100 documents from 20 Newsgroups and runs the full pipeline.
Verifies output format and basic sanity properties.
"""

import pytest
from src.pipeline import run_pipeline
from src.utils import create_spark_session, load_newsgroups, preprocess_text
from src.verification import true_jaccard


@pytest.fixture(scope="module")
def spark():
    """Shared Spark session for pipeline tests."""
    session = create_spark_session("TestPipeline")
    yield session
    session.stop()


@pytest.fixture(scope="module")
def small_docs(spark):
    """Load 100 preprocessed 20 Newsgroups documents as an RDD."""
    docs = load_newsgroups()[:100]
    processed = [(doc_id, preprocess_text(text)) for doc_id, text in docs]
    return spark.sparkContext.parallelize(processed)


class TestRunPipeline:
    def test_output_is_dict_with_expected_keys(self, small_docs):
        """run_pipeline must return a dict with all required keys."""
        result = run_pipeline(
            docs_rdd=small_docs,
            k=5,
            num_hashes=64,
            num_bands=16,
            rows_per_band=4,
            threshold=0.5,
        )
        required_keys = {
            "confirmed_pairs",
            "num_candidates",
            "num_confirmed",
            "runtime_shingling",
            "runtime_minhash",
            "runtime_lsh",
            "runtime_verification",
            "runtime_total",
        }
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - result.keys()}"
        )

    def test_confirmed_pairs_are_triples(self, small_docs):
        """Each confirmed pair must be a (int, int, float) triple."""
        result = run_pipeline(
            docs_rdd=small_docs,
            k=5,
            num_hashes=64,
            num_bands=16,
            rows_per_band=4,
            threshold=0.5,
        )
        for triple in result["confirmed_pairs"]:
            assert len(triple) == 3, f"Expected 3-tuple, got {triple}"
            doc_a, doc_b, score = triple
            assert isinstance(doc_a, int)
            assert isinstance(doc_b, int)
            assert isinstance(score, float)

    def test_all_confirmed_pairs_above_threshold(self, small_docs):
        """All confirmed pairs must have Jaccard >= threshold."""
        threshold = 0.5
        result = run_pipeline(
            docs_rdd=small_docs,
            k=5,
            num_hashes=64,
            num_bands=16,
            rows_per_band=4,
            threshold=threshold,
        )
        for _, _, score in result["confirmed_pairs"]:
            assert score >= threshold, (
                f"Pair with score {score:.3f} below threshold {threshold}"
            )

    def test_pair_ids_are_normalized(self, small_docs):
        """All confirmed pair IDs must satisfy doc_id_a < doc_id_b."""
        result = run_pipeline(
            docs_rdd=small_docs,
            k=5,
            num_hashes=64,
            num_bands=16,
            rows_per_band=4,
            threshold=0.5,
        )
        for doc_a, doc_b, _ in result["confirmed_pairs"]:
            assert doc_a < doc_b, f"Pair not normalized: ({doc_a}, {doc_b})"

    def test_num_confirmed_matches_list_length(self, small_docs):
        """num_confirmed must equal the length of confirmed_pairs."""
        result = run_pipeline(
            docs_rdd=small_docs,
            k=5,
            num_hashes=64,
            num_bands=16,
            rows_per_band=4,
            threshold=0.5,
        )
        assert result["num_confirmed"] == len(result["confirmed_pairs"])

    def test_runtimes_are_positive(self, small_docs):
        """All timing fields must be positive floats."""
        result = run_pipeline(
            docs_rdd=small_docs,
            k=5,
            num_hashes=64,
            num_bands=16,
            rows_per_band=4,
            threshold=0.5,
        )
        for key in ["runtime_shingling", "runtime_minhash", "runtime_lsh",
                    "runtime_verification", "runtime_total"]:
            assert result[key] > 0, f"{key} should be positive, got {result[key]}"


class TestTrueJaccard:
    """Basic unit tests for the true_jaccard helper."""

    def test_identical_sets(self):
        s = {1, 2, 3, 4}
        assert true_jaccard(s, s) == 1.0

    def test_disjoint_sets(self):
        assert true_jaccard({1, 2}, {3, 4}) == 0.0

    def test_known_value(self):
        """J({1,2,3}, {2,3,4}) = |{2,3}| / |{1,2,3,4}| = 0.5"""
        assert true_jaccard({1, 2, 3}, {2, 3, 4}) == 0.5

    def test_empty_sets(self):
        assert true_jaccard(set(), set()) == 0.0

    def test_one_empty(self):
        assert true_jaccard({1, 2, 3}, set()) == 0.0

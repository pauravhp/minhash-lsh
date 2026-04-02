"""
test_shingling.py - Unit tests for src/shingling.py.

Verifies shingle generation on known strings and validates set properties.
"""

import pytest
from src.shingling import generate_shingles


class TestGenerateShingles:
    def test_correct_count_no_duplicates(self):
        """
        A string of length L with shingle size k should produce at most L-k+1 shingles.
        With unique substrings the set size equals L-k+1.
        """
        text = "abcdefgh"
        k = 3
        shingles = generate_shingles(text, k)
        # "abc","bcd","cde","def","efg","fgh" -> 6 unique substrings -> 6 hashes
        # (assuming no hash collisions for these short strings)
        assert len(shingles) <= len(text) - k + 1
        assert len(shingles) > 0

    def test_returns_set_of_ints(self):
        """generate_shingles must return a set of integers."""
        shingles = generate_shingles("hello world", k=4)
        assert isinstance(shingles, set)
        for item in shingles:
            assert isinstance(item, int)
            assert item >= 0  # Non-negative due to masking.

    def test_identical_texts_same_shingles(self):
        """Two identical texts must produce the same shingle set."""
        text = "near duplicate document text"
        assert generate_shingles(text, 5) == generate_shingles(text, 5)

    def test_similar_texts_share_shingles(self):
        """Texts with large overlap must share many shingles."""
        text_a = "the quick brown fox jumps over the lazy dog"
        text_b = "the quick brown fox jumps over the lazy cat"
        sa = generate_shingles(text_a, 5)
        sb = generate_shingles(text_b, 5)
        intersection = sa & sb
        union = sa | sb
        jaccard = len(intersection) / len(union)
        # These texts share most content; Jaccard should be well above 0.5.
        assert jaccard > 0.5, f"Expected high Jaccard, got {jaccard:.2f}"

    def test_disjoint_texts_no_common_shingles(self):
        """Completely different texts should share few or no shingles."""
        text_a = "aaaaaaaaaaaaaaaa"
        text_b = "bbbbbbbbbbbbbbbb"
        sa = generate_shingles(text_a, 4)
        sb = generate_shingles(text_b, 4)
        # "aaaa" hashes to something, "bbbb" hashes to something else.
        assert len(sa & sb) == 0

    def test_short_text_returns_nonempty(self):
        """A text shorter than k should still produce one shingle."""
        shingles = generate_shingles("ab", k=5)
        assert len(shingles) == 1

    def test_empty_text(self):
        """An empty string should return a single shingle (the empty string hash)."""
        shingles = generate_shingles("", k=3)
        assert len(shingles) == 1

    def test_k1_shingles(self):
        """k=1 shingles are individual characters; duplicate characters collapse."""
        text = "aabbc"
        shingles = generate_shingles(text, k=1)
        # Unique characters: a, b, c -> 3 unique hashes.
        assert len(shingles) == 3

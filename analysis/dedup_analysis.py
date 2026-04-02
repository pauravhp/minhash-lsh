"""
dedup_analysis.py - Deduplication audit of the 20 Newsgroups dataset.

Mirrors the analysis performed by Lee et al. (2022) on C4 and The Pile,
measuring how much near-duplicate content exists in the dataset even after
standard preprocessing.

Lee et al. (2022): "Deduplicating Training Data Makes Language Models Better."
ACL 2022. Found that C4 contains many near-duplicate documents, and that
deduplication reduces memorization by 10x and improves perplexity.

Usage:
    python analysis/dedup_analysis.py

Output:
    results/dedup_analysis_summary.txt
"""

import logging
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import run_pipeline
from src.utils import create_spark_session, load_newsgroups, preprocess_text

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
    "k": 5,
    "num_hashes": 128,
    "num_bands": 16,
    "rows_per_band": 8,
    "similarity_threshold": 0.5,
    "output_file": "results/dedup_analysis_summary.txt",
}
# ============================================================


def analyze_duplicate_distribution(confirmed_pairs: list, total_docs: int) -> dict:
    """
    Compute statistics about the near-duplicate distribution.

    Args:
        confirmed_pairs: List of (doc_id_a, doc_id_b, jaccard) tuples.
        total_docs: Total number of documents in the corpus.

    Returns:
        Dict with duplicate stats.
    """
    # Count how many pairs each document appears in.
    doc_pair_count = defaultdict(int)
    for doc_a, doc_b, _ in confirmed_pairs:
        doc_pair_count[doc_a] += 1
        doc_pair_count[doc_b] += 1

    docs_with_duplicate = len(doc_pair_count)
    pct_with_duplicate = 100.0 * docs_with_duplicate / total_docs if total_docs > 0 else 0.0

    jaccard_scores = [j for _, _, j in confirmed_pairs]
    mean_j = sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0.0

    # Histogram bins.
    bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    hist = {}
    for lo, hi in bins:
        label = f"[{lo:.1f}, {hi:.1f})"
        hist[label] = sum(1 for j in jaccard_scores if lo <= j < hi)

    return {
        "total_docs": total_docs,
        "total_confirmed_pairs": len(confirmed_pairs),
        "docs_with_at_least_one_duplicate": docs_with_duplicate,
        "pct_docs_with_duplicate": round(pct_with_duplicate, 2),
        "mean_jaccard_among_pairs": round(mean_j, 4),
        "jaccard_histogram": hist,
    }


def write_summary(stats: dict, result: dict, config: dict, out_path: str) -> None:
    """
    Write a human-readable deduplication summary to a text file.

    Args:
        stats: Output of analyze_duplicate_distribution.
        result: Pipeline result dict.
        config: Experiment config dict.
        out_path: Output file path.
    """
    lines = [
        "Near-Duplicate Content Analysis of 20 Newsgroups",
        "=" * 60,
        "",
        "Motivation:",
        "  This analysis mirrors the deduplication audit performed by Lee et al.",
        "  (2022) on C4 and The Pile, where they found significant near-duplicate",
        "  content even in supposedly curated datasets. Deduplication was shown to",
        "  reduce model memorization by 10x and improve perplexity on downstream",
        "  language modeling tasks.",
        "",
        "Pipeline Configuration:",
        f"  Shingle size k   : {config['k']}",
        f"  Signature length n: {config['num_hashes']}",
        f"  LSH bands b      : {config['num_bands']}",
        f"  LSH rows r       : {config['rows_per_band']}",
        f"  Similarity threshold: {config['similarity_threshold']}",
        "",
        "Results:",
        f"  Total documents analyzed   : {stats['total_docs']}",
        f"  Total confirmed pairs      : {stats['total_confirmed_pairs']}",
        f"  Docs with >= 1 near-dup    : {stats['docs_with_at_least_one_duplicate']} "
        f"({stats['pct_docs_with_duplicate']:.1f}%)",
        f"  Mean Jaccard among pairs   : {stats['mean_jaccard_among_pairs']:.4f}",
        "",
        "Per-Phase Runtimes:",
        f"  Shingling                  : {result['runtime_shingling']:.1f}s",
        f"  MinHash signature          : {result['runtime_minhash']:.1f}s",
        f"  LSH candidate generation   : {result['runtime_lsh']:.1f}s",
        f"  Verification               : {result['runtime_verification']:.1f}s",
        f"  Total                      : {result['runtime_total']:.1f}s",
        "",
        "Jaccard Similarity Distribution (among confirmed pairs):",
    ]

    for bucket, count in stats["jaccard_histogram"].items():
        pct = 100.0 * count / stats["total_confirmed_pairs"] if stats["total_confirmed_pairs"] else 0
        lines.append(f"  {bucket}: {count:5d} pairs ({pct:.1f}%)")

    lines += [
        "",
        "Interpretation:",
        "  The above results show that even in 20 Newsgroups, a relatively small",
        "  corpus (~18,000 documents), a non-trivial fraction of documents have",
        "  near-duplicate counterparts. This is consistent with the findings of",
        "  Lee et al. (2022), who observed that C4 (a large web crawl corpus)",
        "  contains many near-duplicates from repeated web scraping of the same",
        "  pages. For language model training corpora orders of magnitude larger,",
        "  the MinHash-LSH approach is essential since brute-force O(N^2) Jaccard",
        "  computation would be computationally prohibitive.",
        "",
        "Reference:",
        "  Lee, K., Ippolito, D., Nystrom, A., Zhang, C., Eck, D., Callison-Burch, C.,",
        "  & Carlini, N. (2022). Deduplicating Training Data Makes Language Models",
        "  Better. Proceedings of ACL 2022.",
    ]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Summary written to %s", out_path)


def main():
    """Run the deduplication analysis on the full 20 Newsgroups dataset."""
    os.makedirs("results", exist_ok=True)

    logger.info("=" * 60)
    logger.info("Deduplication Analysis (Lee et al. 2022 inspired)")
    logger.info("=" * 60)

    spark = create_spark_session("DedupAnalysis")
    sc = spark.sparkContext

    # Load all documents.
    logger.info("Loading 20 Newsgroups dataset...")
    all_docs = load_newsgroups()
    total_docs = len(all_docs)
    logger.info("Loaded %d documents.", total_docs)

    processed = [(doc_id, preprocess_text(text)) for doc_id, text in all_docs]
    docs_rdd = sc.parallelize(processed, numSlices=8)

    # Run full pipeline.
    logger.info("Running MinHash-LSH pipeline...")
    result = run_pipeline(
        docs_rdd=docs_rdd,
        k=CONFIG["k"],
        num_hashes=CONFIG["num_hashes"],
        num_bands=CONFIG["num_bands"],
        rows_per_band=CONFIG["rows_per_band"],
        threshold=CONFIG["similarity_threshold"],
    )

    # Analyze.
    stats = analyze_duplicate_distribution(result["confirmed_pairs"], total_docs)

    logger.info("Docs with >= 1 near-duplicate: %d (%.1f%%)",
                stats["docs_with_at_least_one_duplicate"],
                stats["pct_docs_with_duplicate"])
    logger.info("Total confirmed pairs: %d", stats["total_confirmed_pairs"])

    write_summary(stats, result, CONFIG, CONFIG["output_file"])

    spark.stop()
    logger.info("Analysis complete.")


if __name__ == "__main__":
    main()

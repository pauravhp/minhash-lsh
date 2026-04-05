# Near-Duplicate Document Detection with MinHash and LSH

A PySpark implementation of the MinHash + Locality-Sensitive Hashing (LSH)
pipeline for near-duplicate document detection, built for CSC 502 (Systems
for Massive Datasets) at the University of Victoria.

The pipeline detects near-duplicate documents in the 20 Newsgroups corpus
(~18,000 documents) using a three-phase MapReduce approach: character
k-shingling, MinHash signature generation, and LSH banding for candidate
pair generation, followed by exact Jaccard verification.

## Papers

1. **Broder, A. Z. (1997).** "On the Resemblance and Containment of Documents."
   IEEE SEQUENCES 1997.
   Introduced MinHash for estimating Jaccard similarity via min-wise independent
   permutations. Key result: P[min(h(A)) = min(h(B))] = J(A, B).

2. **Lee, K., et al. (2022).** "Deduplicating Training Data Makes Language Models
   Better." ACL 2022.
   Applied MinHash deduplication to C4 and The Pile, finding that deduplication
   reduces memorization by 10x and improves language model perplexity.

## Algorithm Overview

```
Documents -> [k-Shingling] -> Shingle Sets -> [MinHash] -> Signatures
         -> [LSH Banding] -> Candidate Pairs -> [Verification] -> Confirmed Pairs
```

1. **Shingling (Phase 1):** Each document is converted to a set of character
   k-shingles (substrings of length k), each hashed to a 32-bit integer.
   Embarrassingly parallel (no inter-document communication).

2. **MinHash (Phase 1, continued):** For each document, compute a signature
   vector of length n using a universal hash family h_i(x) = (a_i * x + b_i) mod p.
   By Broder (1997), the probability that two documents agree at any signature
   position equals their Jaccard similarity.

3. **LSH Banding (Phase 2):** Divide each signature into b bands of r rows.
   Two documents become candidates if their signatures match in all rows of
   at least one band. Detection probability: P = 1 - (1 - s^r)^b.
   Uses PySpark groupByKey to collect collision buckets.

4. **Verification (Phase 3):** For each candidate pair, compute the exact
   Jaccard similarity from the original shingle sets and filter below threshold.

## Requirements

- Python 3.9+
- Java 8+ (for PySpark; Java 11 or 17 recommended)
- pip

## Setup

```bash
# 1. Clone the repository and enter the project directory.
cd csc502-minhash-lsh

# 2. Create a virtual environment.
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies.
pip install -r requirements.txt

# 4. On macOS, if you see SSL certificate errors when downloading the dataset,
#    install certifi and set the cert file:
pip install certifi
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")

# 5. Pre-download the 20 Newsgroups dataset (required before running tests).
python -c "from sklearn.datasets import fetch_20newsgroups; fetch_20newsgroups(subset='all', remove=('headers','footers','quotes'))"
```

## Running Tests

```bash
source venv/bin/activate
python -m pytest tests/ -v
```

All tests should pass in under 30 seconds total.

## Running Experiments

Run each experiment script manually. Results are written to `results/`.

```bash
source venv/bin/activate

# Experiment 1 (Priority): LSH parameter sensitivity
python experiments/run_param_sensitivity.py

# Experiment 2 (Priority): Scalability vs. dataset size
python experiments/run_scalability.py

# Experiment 3 (Stretch): MinHash estimation error vs. signature length
python experiments/run_signature_length.py

# Experiment 4 (Stretch): Qualitative near-duplicate examples
python experiments/run_qualitative.py

# Lee et al. (2022) inspired deduplication audit
python analysis/dedup_analysis.py
```

## Generating Figures

After running experiments, generate all figures from the CSVs:

```bash
python experiments/generate_figures.py
```

Figures are written to `results/figures/`.

## Project Structure

```
csc502-minhash-lsh/
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── src/
│   ├── __init__.py
│   ├── utils.py                     # Data loading, preprocessing, Spark session
│   ├── shingling.py                 # Character k-shingle generation
│   ├── minhash.py                   # MinHash signature computation
│   ├── lsh.py                       # LSH banding and candidate generation
│   ├── verification.py              # True Jaccard computation and filtering
│   └── pipeline.py                  # End-to-end pipeline orchestration
├── experiments/
│   ├── run_param_sensitivity.py     # Experiment 1: (b,r) sensitivity
│   ├── run_scalability.py           # Experiment 2: runtime vs. dataset size
│   ├── run_signature_length.py      # Experiment 3 (stretch): n vs. error
│   ├── run_qualitative.py           # Experiment 4 (stretch): example pairs
│   └── generate_figures.py          # Reads CSVs, writes PNGs
├── results/
│   ├── (CSVs written by experiment scripts)
│   └── figures/
│       └── (PNGs written by generate_figures.py)
├── tests/
│   ├── test_shingling.py
│   ├── test_minhash.py
│   ├── test_lsh.py
│   └── test_pipeline.py
├── analysis/
│   └── dedup_analysis.py            # Lee et al. (2022) deduplication audit
└── docs/
    └── RESULTS_SUMMARY.md           # Template for experiment results
```

## Performance Notes

- Full pipeline on 18,000 documents targets under 15 minutes on a Mac M1 (16 GB).
- PySpark runs in `local[*]` mode using all available cores.
- Brute-force baseline is O(N^2); only run on subsets up to 5,000 documents.
- Driver memory is set to 10 GB to leave headroom for the OS.

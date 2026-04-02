"""
generate_figures.py - Read experiment CSVs and produce matplotlib figures.

Reads from results/ and writes PNGs to results/figures/.
Can be run standalone at any time after experiments complete:

    python experiments/generate_figures.py

Figures generated:
    - param_sensitivity.png: Precision/Recall/F1 vs. (b,r) + S-curve overlay
    - scalability.png: Runtime vs. num_docs for LSH vs. brute-force
    - signature_length.png: MAE vs. n with theoretical 1/sqrt(n) overlay (if data exists)
"""

import csv
import math
import os
import sys

# Add project root to path so we can import src modules for S-curve computation.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def read_csv(filepath: str) -> list:
    """
    Read a CSV file and return a list of dicts.

    Args:
        filepath: Path to the CSV file.

    Returns:
        List of row dicts, or empty list if file does not exist.
    """
    if not os.path.exists(filepath):
        print(f"[WARN] {filepath} not found, skipping.")
        return []
    with open(filepath, newline="") as f:
        return list(csv.DictReader(f))


def plot_param_sensitivity(rows: list, out_path: str) -> None:
    """
    Plot Experiment 1 results: precision/recall/F1 bars + theoretical S-curves.

    Args:
        rows: List of row dicts from param_sensitivity.csv.
        out_path: Path to save the PNG.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [f"b={r['b']}\nr={r['r']}" for r in rows]
    precisions = [float(r["precision"]) for r in rows]
    recalls = [float(r["recall"]) for r in rows]
    f1s = [float(r["f1"]) for r in rows]

    x = np.arange(len(labels))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: bar chart of P/R/F1.
    ax = axes[0]
    ax.bar(x - width, precisions, width, label="Precision", color="#4C72B0")
    ax.bar(x, recalls, width, label="Recall", color="#DD8452")
    ax.bar(x + width, f1s, width, label="F1", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("(b, r) Configuration")
    ax.set_ylabel("Score")
    ax.set_title("Experiment 1: LSH Parameter Sensitivity\nPrecision / Recall / F1")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Right: theoretical S-curves.
    ax2 = axes[1]
    s_values = np.linspace(0, 1, 200)
    colors = plt.cm.tab10(np.linspace(0, 1, len(rows)))
    for row, color in zip(rows, colors):
        b, r = int(row["b"]), int(row["r"])
        probs = [1.0 - (1.0 - s ** r) ** b for s in s_values]
        ax2.plot(s_values, probs, label=f"b={b}, r={r}", color=color)

    ax2.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="threshold=0.5")
    ax2.set_xlabel("Jaccard Similarity")
    ax2.set_ylabel("P(candidate)")
    ax2.set_title("Theoretical S-Curves\nP(candidate) = 1 - (1 - s^r)^b")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_scalability(rows: list, out_path: str) -> None:
    """
    Plot Experiment 2 results: runtime vs. num_docs for LSH and brute-force.

    Args:
        rows: List of row dicts from scalability.csv.
        out_path: Path to save the PNG.
    """
    import matplotlib.pyplot as plt

    lsh_rows = [r for r in rows if r["method"] == "lsh"]
    bf_rows = [r for r in rows if r["method"] == "brute_force"]

    lsh_x = [int(r["num_docs"]) for r in lsh_rows]
    lsh_y = [float(r["runtime_seconds"]) for r in lsh_rows]
    bf_x = [int(r["num_docs"]) for r in bf_rows]
    bf_y = [float(r["runtime_seconds"]) for r in bf_rows]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: runtime vs. num_docs.
    ax = axes[0]
    ax.plot(lsh_x, lsh_y, "o-", color="#4C72B0", label="LSH Pipeline", linewidth=2)
    if bf_x:
        ax.plot(bf_x, bf_y, "s--", color="#C44E52", label="Brute Force", linewidth=2)
    ax.set_xlabel("Number of Documents")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Experiment 2: Scalability\nRuntime vs. Dataset Size")
    ax.legend()
    ax.grid(alpha=0.3)

    # Right: pairs found vs. num_docs.
    ax2 = axes[1]
    lsh_pairs = [int(r["num_pairs_found"]) for r in lsh_rows]
    ax2.plot(lsh_x, lsh_pairs, "o-", color="#4C72B0", label="LSH Candidates", linewidth=2)
    if bf_rows:
        bf_pairs = [int(r["num_pairs_found"]) for r in bf_rows]
        ax2.plot(bf_x, bf_pairs, "s--", color="#C44E52",
                 label="Brute Force Pairs", linewidth=2)
    ax2.set_xlabel("Number of Documents")
    ax2.set_ylabel("Pairs Found")
    ax2.set_title("Pairs Found vs. Dataset Size")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_signature_length(rows: list, out_path: str) -> None:
    """
    Plot Experiment 3 results: MAE vs. signature length n with theory overlay.

    Args:
        rows: List of row dicts from signature_length.csv.
        out_path: Path to save the PNG.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    n_values = [int(r["n"]) for r in rows]
    mae_values = [float(r["mean_absolute_error"]) for r in rows]
    std_values = [float(r["std_error"]) for r in rows]
    theory = [1.0 / math.sqrt(n) for n in n_values]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(n_values, mae_values, yerr=std_values,
                fmt="o-", color="#4C72B0", label="Empirical MAE +/- std", linewidth=2)
    ax.plot(n_values, theory, "r--", label="Theoretical: 1/sqrt(n)", linewidth=2)

    ax.set_xlabel("Signature Length (n)")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title(
        "Experiment 3: MinHash Estimation Error vs. Signature Length\n"
        "Broder (1997): Expected error decreases as 1/sqrt(n)"
    )
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    """Read all result CSVs and generate figures."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print(f"Reading results from: {RESULTS_DIR}/")
    print(f"Writing figures to:   {FIGURES_DIR}/")

    # Experiment 1: parameter sensitivity.
    rows_ps = read_csv(os.path.join(RESULTS_DIR, "param_sensitivity.csv"))
    if rows_ps:
        plot_param_sensitivity(rows_ps, os.path.join(FIGURES_DIR, "param_sensitivity.png"))
    else:
        print("No param_sensitivity.csv found. Run run_param_sensitivity.py first.")

    # Experiment 2: scalability.
    rows_sc = read_csv(os.path.join(RESULTS_DIR, "scalability.csv"))
    if rows_sc:
        plot_scalability(rows_sc, os.path.join(FIGURES_DIR, "scalability.png"))
    else:
        print("No scalability.csv found. Run run_scalability.py first.")

    # Experiment 3: signature length (stretch, optional).
    rows_sl = read_csv(os.path.join(RESULTS_DIR, "signature_length.csv"))
    if rows_sl:
        plot_signature_length(rows_sl, os.path.join(FIGURES_DIR, "signature_length.png"))
    else:
        print("No signature_length.csv found (stretch goal, optional).")

    print("Figure generation complete.")


if __name__ == "__main__":
    main()

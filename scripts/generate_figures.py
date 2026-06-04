#!/usr/bin/env python3
"""Generate all figures from experiment results.

Reads JSON result files from results/logs/ and produces publication-quality
figures in results/figures/ for the report.

Usage:
    python scripts/generate_figures.py [--results_dir results/logs] [--save_dir results/figures]
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visualization.plot_utils import set_style
from src.visualization.plot_comparison import (
    load_results,
    plot_accuracy_vs_params,
    plot_sample_efficiency,
    plot_computational_efficiency,
    plot_cross_dataset_summary,
)
from src.visualization.plot_sparsity import (
    plot_sparsity_vs_accuracy,
)


def main():
    parser = argparse.ArgumentParser(description="Generate all figures from experiment results")
    parser.add_argument("--results_dir", default="results/logs")
    parser.add_argument("--save_dir", default="results/figures")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_style()

    print("Loading results...")
    results = load_results(args.results_dir)
    print(f"Loaded {len(results)} result entries")

    if not results:
        print("No results found. Run experiments first.")
        return

    datasets = set(r.get("dataset") for r in results if isinstance(r, dict) and "dataset" in r)
    datasets = [d for d in ["cub200", "flowers102", "stanford_cars"] if d in datasets]
    print(f"Datasets with results: {datasets}")

    # 1. Accuracy vs Params (per dataset)
    print("\nGenerating accuracy vs params plots...")
    for ds in datasets:
        plot_accuracy_vs_params(results, ds, args.save_dir)

    # 2. Sample efficiency (per dataset)
    print("Generating sample efficiency plots...")
    for ds in datasets:
        plot_sample_efficiency(results, ds, args.save_dir)

    # 3. Cross-dataset summary
    print("Generating cross-dataset summary...")
    plot_cross_dataset_summary(results, args.save_dir)

    # 4. Computational efficiency
    print("Generating computational efficiency plots...")
    for ds in datasets:
        plot_computational_efficiency(results, ds, args.save_dir)

    # 5. Sparsity analysis (SSF-Sparse)
    print("Generating sparsity analysis plots...")
    for ds in datasets:
        plot_sparsity_vs_accuracy(results, ds, args.save_dir)

    print(f"\nAll figures saved to {args.save_dir}/")
    print("Figures ready for report.")


if __name__ == "__main__":
    main()

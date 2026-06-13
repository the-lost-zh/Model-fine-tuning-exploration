"""SSF-Sparse specific plots: sparsity curves and heatmaps."""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

from .plot_utils import set_style, get_color, DATASET_LABELS, save_figure


def plot_sparsity_vs_accuracy(sparsity_results, dataset, save_dir="results/figures"):
    """Figure: Accuracy vs. Sparsity Ratio (line plot).

    Shows how accuracy changes as more SSF gates are pruned.
    """
    set_style()
    ds_results = [r for r in sparsity_results
                  if isinstance(r, dict) and r.get("dataset") == dataset]

    if not ds_results:
        print(f"No sparsity results for {dataset}")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Sort by sparsity_lambda
    ds_sorted = sorted(ds_results, key=lambda r: r.get("sparsity_lambda", 0))

    lambdas = [r.get("sparsity_lambda", 0) for r in ds_sorted]
    accs = [r["test_acc"] * 100 for r in ds_sorted]
    sparsities = [r.get("mean_sparsity", 0) * 100 for r in ds_sorted]

    color = get_color("ssf_sparse")
    ax2 = ax.twinx()

    ax.plot(lambdas, accs, marker="o", color=color, linewidth=2, markersize=8,
            label="Accuracy")
    ax2.plot(lambdas, sparsities, marker="s", color="#e377c2", linewidth=2,
             markersize=8, linestyle="--", label="Sparsity %")

    ax.set_xlabel("Sparsity Lambda (λ)")
    ax.set_ylabel("Top-1 Accuracy (%)", color=color)
    ax2.set_ylabel("Channels Pruned (%)", color="#e377c2")
    ax.set_title(f"SSF-Sparse: Accuracy vs. Sparsity on {DATASET_LABELS.get(dataset, dataset)}")

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    save_path = os.path.join(save_dir, f"sparsity_vs_accuracy_{dataset}.png")
    save_figure(fig, save_path)
    return save_path


def plot_sparsity_heatmap(sparsity_stats, dataset, save_dir="results/figures"):
    """Figure: Per-layer sparsity heatmap.

    Rows = layer index (0-11), Columns = component (qkv/proj/fc1/fc2/norm1/norm2)
    Cell value = fraction of active gates (g > 0.5).
    """
    set_style()
    if not sparsity_stats:
        return

    # Build heatmap data: layers × components
    components = ["qkv", "proj", "fc1", "fc2", "norm1", "norm2"]
    n_layers = max(s["layer"] for s in sparsity_stats) + 1
    heatmap = np.zeros((n_layers, len(components)))

    for s in sparsity_stats:
        layer = s["layer"]
        comp = s["op"]
        if comp in components:
            heatmap[layer, components.index(comp)] = s["frac_active"]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(heatmap, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(components)))
    ax.set_xticklabels(components, rotation=45, ha="right")
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"Layer {i+1}" for i in range(n_layers)])
    ax.set_xlabel("Component")
    ax.set_ylabel("Layer")
    ax.set_title(f"SSF-Sparse: Fraction of Active Gates\n{DATASET_LABELS.get(dataset, dataset)}")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Fraction Active (g > 0.5)")

    # Annotate cells
    for i in range(n_layers):
        for j in range(len(components)):
            ax.text(j, i, f"{heatmap[i, j]:.2f}", ha="center", va="center",
                    fontsize=8)

    save_path = os.path.join(save_dir, f"sparsity_heatmap_{dataset}.png")
    save_figure(fig, save_path)
    return save_path


def plot_gate_distribution(sparsity_stats, dataset, save_dir="results/figures"):
    """Figure: Histogram of gate values across all SSF positions."""
    set_style()
    if not sparsity_stats:
        return

    all_gates = [s["mean_gate"] for s in sparsity_stats]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(all_gates, bins=30, color=get_color("ssf_sparse"), edgecolor="white",
            alpha=0.8)
    ax.axvline(x=0.5, color="red", linestyle="--", linewidth=1, label="g = 0.5")
    ax.set_xlabel("Mean Gate Value")
    ax.set_ylabel("Count")
    ax.set_title(f"SSF-Sparse: Gate Value Distribution\n{DATASET_LABELS.get(dataset, dataset)}")
    ax.legend()

    save_path = os.path.join(save_dir, f"gate_distribution_{dataset}.png")
    save_figure(fig, save_path)
    return save_path

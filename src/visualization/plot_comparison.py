"""Generate comparison plots: bar charts, scatter plots, heatmaps."""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

from .plot_utils import set_style, get_color, get_label, DATASET_LABELS, save_figure


def load_results(results_dir="results/logs"):
    """Load all experiment results from JSON files.

    Returns:
        list of result dicts
    """
    results = []
    for fname in os.listdir(results_dir):
        if fname.endswith(".json"):
            with open(os.path.join(results_dir, fname)) as f:
                data = json.load(f)
                if isinstance(data, list):
                    results.extend(data)
                else:
                    results.append(data)
    return results


def plot_accuracy_vs_params(results, dataset, save_dir="results/figures"):
    """Figure: Accuracy vs. Trainable Parameters (bar chart with param annotation).

    This is the main comparison figure for a single dataset.
    """
    set_style()
    ds_results = [r for r in results if r.get("dataset") == dataset]
    if not ds_results:
        print(f"No results for dataset {dataset}")
        return

    # Aggregate by method (average over seeds)
    methods = ["linear_probe", "full_ft", "bitfit", "lora", "ssf", "adaptformer"]
    accs = []
    params = []
    labels = []
    colors = []

    for m in methods:
        m_results = [r for r in ds_results if r["method"] == m]
        if not m_results:
            continue
        avg_acc = np.mean([r["test_acc"] for r in m_results]) * 100
        avg_params = np.mean([r.get("trainable_params", 0) for r in m_results])
        accs.append(avg_acc)
        params.append(avg_params)
        labels.append(get_label(m))
        colors.append(get_color(m))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, accs, color=colors, edgecolor="white", linewidth=0.8)

    # Annotate with param count
    for i, (bar, p) in enumerate(zip(bars, params)):
        if p > 1e6:
            text = f"{p/1e6:.1f}M"
        else:
            text = f"{p/1e3:.1f}K"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                text, ha="center", va="bottom", fontsize=9, color="#555555")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title(f"Accuracy vs. Trainable Parameters on {DATASET_LABELS.get(dataset, dataset)}")
    ax.set_ylim(bottom=min(accs) - 5, top=max(accs) + 5)

    save_path = os.path.join(save_dir, f"accuracy_vs_params_{dataset}.png")
    save_figure(fig, save_path)
    return save_path


def plot_sample_efficiency(results, dataset, save_dir="results/figures"):
    """Figure: Accuracy vs. Training Data Fraction (line plot).

    Shows how each method degrades with less training data.
    """
    set_style()
    ds_results = [r for r in results if isinstance(r, dict) and r.get("dataset") == dataset
                  and "fraction" in r]
    if not ds_results:
        print(f"No sample efficiency results for {dataset}")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    methods_seen = set()

    for r in ds_results:
        m = r["method"]
        if m in methods_seen:
            continue
        methods_seen.add(m)
        m_results = sorted(
            [x for x in ds_results if x["method"] == m],
            key=lambda x: x["fraction"],
        )
        fracs = [x["fraction"] for x in m_results]
        accs = [x["test_acc"] * 100 for x in m_results]
        ax.plot(fracs, accs, marker="o", color=get_color(m), label=get_label(m),
                linewidth=2, markersize=6)

    ax.set_xlabel("Training Data Fraction")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title(f"Sample Efficiency on {DATASET_LABELS.get(dataset, dataset)}")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1.05)

    save_path = os.path.join(save_dir, f"sample_efficiency_{dataset}.png")
    save_figure(fig, save_path)
    return save_path


def plot_computational_efficiency(results, dataset, save_dir="results/figures"):
    """Figure: Accuracy vs. Training Time (scatter plot).

    Each point = one method, sized by parameter count.
    """
    set_style()
    ds_results = [r for r in results if isinstance(r, dict) and r.get("dataset") == dataset
                  and "train_time" in r]
    if not ds_results:
        print(f"No training time results for {dataset}")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for r in ds_results:
        m = r["method"]
        avg_acc = np.mean([x["test_acc"] for x in ds_results if x["method"] == m]) * 100 if any(
            x["method"] == m for x in ds_results) else r["test_acc"] * 100
        avg_time = r.get("train_time", 0) / 60  # minutes
        ax.scatter(avg_time, avg_acc, c=get_color(m), label=get_label(m),
                   s=150, edgecolors="white", linewidth=0.5, zorder=3)

    ax.set_xlabel("Training Time (minutes)")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title(f"Accuracy vs. Training Time on {DATASET_LABELS.get(dataset, dataset)}")
    ax.legend(loc="lower right")

    save_path = os.path.join(save_dir, f"compute_efficiency_{dataset}.png")
    save_figure(fig, save_path)
    return save_path


def plot_layer_ablation_heatmap(results, dataset, method, save_dir="results/figures"):
    """Figure: Heatmap of accuracy when removing adaptation from layer groups.

    Rows = component type (attn only, mlp only, both), Columns = layer groups.
    """
    set_style()
    ds_results = [r for r in results if isinstance(r, dict)
                  and r.get("dataset") == dataset
                  and r.get("method") == method
                  and "group" in r]

    if not ds_results:
        print(f"No layer ablation results for {method} on {dataset}")
        return

    # Get baseline (all layers)
    baseline = [r for r in ds_results if r["group"] == "all"]
    baseline_acc = baseline[0]["test_acc"] * 100 if baseline else 0

    groups = ["early", "middle", "late"]
    acc_drops = []
    for group in groups:
        g = [r for r in ds_results if r["group"] == group]
        if g:
            acc_drops.append(baseline_acc - g[0]["test_acc"] * 100)
        else:
            acc_drops.append(0)

    fig, ax = plt.subplots(figsize=(6, 3))
    x = np.arange(len(groups))
    bars = ax.bar(x, acc_drops, color=[get_color(method)] * len(groups),
                  edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(["Early (1-4)", "Middle (5-8)", "Late (9-12)"])
    ax.set_ylabel("Accuracy Drop (%)")
    ax.set_title(f"Layer-wise Ablation: {get_label(method)} on {DATASET_LABELS.get(dataset, dataset)}")
    ax.axhline(y=0, color="black", linewidth=0.5)

    for bar, drop in zip(bars, acc_drops):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{drop:.1f}%", ha="center", fontsize=10)

    save_path = os.path.join(save_dir, f"layer_ablation_{method}_{dataset}.png")
    save_figure(fig, save_path)
    return save_path


def plot_cross_dataset_summary(results, save_dir="results/figures"):
    """Figure: Grouped bar chart showing each method's accuracy across datasets."""
    set_style()
    datasets = ["cub200", "flowers102", "stanford_cars"]
    methods = ["bitfit", "lora", "ssf", "adaptformer", "full_ft"]

    # Aggregate per dataset per method
    data = {}
    for ds in datasets:
        data[ds] = {}
        for m in methods:
            m_results = [r for r in results if isinstance(r, dict)
                         and r.get("dataset") == ds and r["method"] == m]
            if m_results:
                data[ds][m] = np.mean([r["test_acc"] for r in m_results]) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(datasets))
    width = 0.15
    n = len(methods)

    for i, m in enumerate(methods):
        accs = [data[ds].get(m, 0) for ds in datasets]
        offset = (i - n / 2 + 0.5) * width
        ax.bar(x + offset, accs, width, color=get_color(m), label=get_label(m),
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in datasets], rotation=15, ha="right")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("Cross-Dataset Method Comparison")
    ax.legend(loc="upper right", ncol=2)

    save_path = os.path.join(save_dir, "cross_dataset_summary.png")
    save_figure(fig, save_path)
    return save_path

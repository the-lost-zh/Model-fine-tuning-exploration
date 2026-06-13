"""Plotting utilities: consistent styling, color maps, save helpers."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# Color scheme for methods (consistent across all plots)
METHOD_COLORS = {
    "full_ft": "#d62728",       # red
    "linear_probe": "#7f7f7f",  # gray
    "bitfit": "#ff7f0e",        # orange
    "lora": "#1f77b4",          # blue
    "ssf": "#2ca02c",           # green
    "adaptformer": "#9467bd",   # purple
    "ssf_sparse": "#17becf",    # cyan
}

METHOD_LABELS = {
    "full_ft": "Full FT",
    "linear_probe": "Linear Probe",
    "bitfit": "BitFit",
    "lora": "LoRA",
    "ssf": "SSF",
    "adaptformer": "AdaptFormer",
    "ssf_sparse": "SSF-Sparse",
}

DATASET_LABELS = {
    "cub200": "CUB-200-2011",
    "flowers102": "Oxford Flowers-102",
    "stanford_cars": "Stanford Cars",
}


def set_style():
    """Apply consistent matplotlib style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    matplotlib.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.figsize": (8, 5),
    })


def get_color(method):
    return METHOD_COLORS.get(method, "#333333")


def get_label(method):
    return METHOD_LABELS.get(method, method)


def save_figure(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

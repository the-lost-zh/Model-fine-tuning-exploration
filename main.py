#!/usr/bin/env python3
"""Main entry point for model fine-tuning exploration experiments.

Usage:
    python main.py --method ssf --dataset cub200
    python main.py --method lora --dataset flowers102 --lr 1e-3
    python main.py --method ssf_sparse --dataset cub200 --sparsity_lambda 1e-5
    python main.py --exp layer_ablation --method ssf --dataset cub200
"""

import os
import warnings
warnings.filterwarnings("ignore")

# Configure proxy and disable SSL verification for GFW environment.
# NOTE: This disables SSL globally — only use in development behind a trusted proxy.
_proxy_url = os.environ.get("HTTPS_PROXY", os.environ.get("https_proxy", ""))
if _proxy_url:
    os.environ.setdefault("HTTP_PROXY", _proxy_url)
    os.environ.setdefault("HTTPS_PROXY", _proxy_url)

import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_original_session_request = requests.Session.request
def _patched_request(self, method, url, **kwargs):
    kwargs.setdefault("verify", False)
    return _original_session_request(self, method, url, **kwargs)
requests.Session.request = _patched_request

import argparse
import yaml
import torch
import random
import numpy as np

from src.models import build_vit
from src.models.lora import apply_lora
from src.models.bitfit import apply_bitfit
from src.models.ssf import apply_ssf
from src.models.ssf_sparse import apply_ssf_sparse
from src.models.adaptformer import apply_adaptformer
from src.data.datamodule import build_dataloaders, get_subset_dataloader
from src.trainers.base_trainer import train_model
from src.trainers.metrics import param_summary


METHOD_APPLY = {
    "full_ft": lambda m, cfg: m,  # train all
    "linear_probe": lambda m, cfg: _apply_linear_probe(m),
    "bitfit": lambda m, cfg: apply_bitfit(m),
    "lora": lambda m, cfg: apply_lora(m, r=cfg.get("lora_r", 8), alpha=cfg.get("lora_alpha", 16.0)),
    "ssf": lambda m, cfg: apply_ssf(m),
    "adaptformer": lambda m, cfg: apply_adaptformer(
        m, d_hat=cfg.get("d_hat", 64), scale=cfg.get("adapt_scale", 0.1)),
    "ssf_sparse": lambda m, cfg: apply_ssf_sparse(m),
}


def _apply_linear_probe(model):
    """Freeze all except classifier head."""
    from src.models.base_vit import freeze_all
    freeze_all(model)
    if hasattr(model, "head"):
        for p in model.head.parameters():
            p.requires_grad = True
    return model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_experiment(method, dataset, data_root="./data", seed=42, extra_cfg=None):
    """Run a single fine-tuning experiment.

    Args:
        method: one of full_ft, linear_probe, bitfit, lora, ssf, adaptformer, ssf_sparse
        dataset: one of cub200, flowers102, stanford_cars
        data_root: root directory for datasets
        seed: random seed
        extra_cfg: dict of additional config overrides

    Returns:
        results dict with all metrics
    """
    # Load config
    with open("configs/base.yaml") as f:
        cfg = yaml.safe_load(f)

    # Merge method-specific config
    method_cfg = cfg["methods"].get(method, {})
    cfg.update(method_cfg)
    if extra_cfg:
        cfg.update(extra_cfg)

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build dataloaders
    train_loader, val_loader, test_loader, num_classes = build_dataloaders(
        dataset, data_root, batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"], input_size=cfg["input_size"],
        val_split=cfg["val_split"],
    )

    # Build model
    model = build_vit(
        model_name=cfg["model_name"], pretrained=cfg["pretrained"],
        num_classes=num_classes,
        pretrained_path=extra_cfg.get("pretrained_path") if extra_cfg else None,
    )

    # Apply method
    if method != "full_ft":
        model = METHOD_APPLY[method](model, cfg)

    # Count parameters
    params = param_summary(model)
    print(f"Method: {method} | Trainable: {params['trainable_params']:,} "
          f"({params['trainable_pct']:.3f}%)")

    # Train
    sparsity_lambda = cfg.get("sparsity_lambda", 0.0)
    save_dir = f"results/checkpoints/{method}_{dataset}_seed{seed}"
    tag = f"{method}_{dataset}_seed{seed}"

    results = train_model(
        model, train_loader, val_loader, test_loader,
        lr=cfg["lr"], weight_decay=cfg["weight_decay"],
        epochs=cfg["epochs"], patience=cfg["patience"],
        device=device, sparsity_lambda=sparsity_lambda,
        save_dir=save_dir, tag=tag,
    )

    results.update(params)
    results["method"] = method
    results["dataset"] = dataset
    results["seed"] = seed

    print(f"Results: Test Acc={results['test_acc']:.4f}, "
          f"Val Acc={results['best_val_acc']:.4f}, "
          f"Time={results['train_time']:.1f}s, "
          f"GPU Mem={results['gpu_memory_mb']:.0f}MB")

    return results


def run_sample_efficiency(method, dataset, data_root="./data", seed=42, extra_cfg=None):
    """Run sample efficiency experiment (Experiment 2)."""
    # Load config
    with open("configs/base.yaml") as f:
        cfg = yaml.safe_load(f)
    method_cfg = cfg["methods"].get(method, {})
    cfg.update(method_cfg)
    if extra_cfg:
        cfg.update(extra_cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fractions = [0.1, 0.25, 0.5, 1.0]
    all_results = []

    for frac in fractions:
        set_seed(seed)
        train_loader, val_loader, test_loader, num_classes = get_subset_dataloader(
            dataset, data_root, frac, batch_size=cfg["batch_size"],
            num_workers=cfg["num_workers"], input_size=cfg["input_size"],
        )

        model = build_vit(
            model_name=cfg["model_name"], pretrained=cfg["pretrained"],
            num_classes=num_classes,
        )
        if method != "full_ft":
            model = METHOD_APPLY[method](model, cfg)

        results = train_model(
            model, train_loader, val_loader, test_loader,
            lr=cfg["lr"], weight_decay=cfg["weight_decay"],
            epochs=cfg["epochs"], patience=cfg["patience"],
            device=device,
        )
        results["fraction"] = frac
        results["method"] = method
        results["dataset"] = dataset
        all_results.append(results)
        print(f"  Fraction {frac:.0%}: Test Acc={results['test_acc']:.4f}")

    return all_results


def run_layer_ablation(method, dataset, data_root="./data", seed=42, extra_cfg=None):
    """Run layer-wise ablation experiment (Experiment 4).

    Tests impact of removing adaptation from specific layer groups.
    """
    with open("configs/base.yaml") as f:
        cfg = yaml.safe_load(f)
    method_cfg = cfg["methods"].get(method, {})
    cfg.update(method_cfg)
    if extra_cfg:
        cfg.update(extra_cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(seed)
    train_loader, val_loader, test_loader, num_classes = build_dataloaders(
        dataset, data_root, batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"], input_size=cfg["input_size"],
    )

    # Configuration: which blocks to keep adaptation in
    # None = all layers (full method)
    # "early" = layers 0-3, "middle" = layers 4-7, "late" = layers 8-11
    # "early+middle", "middle+late", "early+late"
    group_configs = [
        ("all", None),
        ("early", [0, 1, 2, 3]),
        ("middle", [4, 5, 6, 7]),
        ("late", [8, 9, 10, 11]),
    ]

    all_results = []

    for group_name, layer_indices in group_configs:
        set_seed(seed)
        model = build_vit(
            model_name=cfg["model_name"], pretrained=cfg["pretrained"],
            num_classes=num_classes,
        )

        if group_name != "all":
            # Apply method but restrict to specified layers
            model = _apply_method_to_layers(model, method, cfg, layer_indices)
        elif method != "full_ft":
            model = METHOD_APPLY[method](model, cfg)

        results = train_model(
            model, train_loader, val_loader, test_loader,
            lr=cfg["lr"], weight_decay=cfg["weight_decay"],
            epochs=cfg["epochs"], patience=cfg["patience"],
            device=device,
        )
        results["group"] = group_name
        results["method"] = method
        results["dataset"] = dataset
        all_results.append(results)
        print(f"  Group {group_name}: Test Acc={results['test_acc']:.4f}")

    return all_results


def _apply_method_to_layers(model, method, cfg, layer_indices):
    """Apply a PEFT method only to specified layer indices.

    This is used for layer ablation experiments.
    """
    from src.models.base_vit import freeze_all

    if method == "ssf":
        freeze_all(model)
        for idx in layer_indices:
            block = model.blocks[idx]
            from src.models.ssf import _insert_ssf_after_op
            if hasattr(block.attn, "qkv"):
                _insert_ssf_after_op(block.attn, "qkv", block.attn.qkv)
            if hasattr(block.attn, "proj"):
                _insert_ssf_after_op(block.attn, "proj", block.attn.proj)
            if hasattr(block.mlp, "fc1"):
                _insert_ssf_after_op(block.mlp, "fc1", block.mlp.fc1)
            if hasattr(block.mlp, "fc2"):
                _insert_ssf_after_op(block.mlp, "fc2", block.mlp.fc2)
    elif method == "lora":
        freeze_all(model)
        for idx in layer_indices:
            from src.models.lora import _replace_linear_with_lora
            block = model.blocks[idx]
            _replace_linear_with_lora(
                block.attn, r=cfg.get("lora_r", 8),
                alpha=cfg.get("lora_alpha", 16.0),
                target_prefixes=["qkv", "proj"],
            )
    elif method == "adaptformer":
        freeze_all(model)
        for idx in layer_indices:
            from src.models.adaptformer import AdaptedMLPBlock
            block = model.blocks[idx]
            d_model = block.mlp.fc1.in_features
            block.mlp = AdaptedMLPBlock(
                block.mlp, d_model,
                d_hat=cfg.get("d_hat", 64),
                scale=cfg.get("adapt_scale", 0.1),
            )
    elif method == "bitfit":
        freeze_all(model)
        for idx in layer_indices:
            for name, param in model.blocks[idx].named_parameters():
                if "bias" in name:
                    param.requires_grad = True

    if hasattr(model, "head"):
        for p in model.head.parameters():
            p.requires_grad = True

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True,
                        choices=["full_ft", "linear_probe", "bitfit", "lora",
                                 "ssf", "adaptformer", "ssf_sparse"])
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["cub200", "flowers102", "stanford_cars"])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--sparsity_lambda", type=float, default=None)
    parser.add_argument("--exp", type=str, default="main",
                        choices=["main", "sample_efficiency", "layer_ablation"])
    parser.add_argument("--pretrained_path", type=str, default=None,
                        help="Local path to pretrained weights (.pth)")
    parser.add_argument("--hf_mirror", action="store_true",
                        help="Use HF mirror (hf-mirror.com) for downloading weights")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results JSON")
    args = parser.parse_args()

    if args.hf_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    extra_cfg = {}
    if args.lr is not None:
        extra_cfg["lr"] = args.lr
    if args.sparsity_lambda is not None:
        extra_cfg["sparsity_lambda"] = args.sparsity_lambda
    if args.pretrained_path is not None:
        extra_cfg["pretrained_path"] = args.pretrained_path

    if args.exp == "main":
        results = run_experiment(args.method, args.dataset, args.data_root,
                                 args.seed, extra_cfg)
    elif args.exp == "sample_efficiency":
        results = run_sample_efficiency(args.method, args.dataset, args.data_root,
                                        args.seed, extra_cfg)
    elif args.exp == "layer_ablation":
        results = run_layer_ablation(args.method, args.dataset, args.data_root,
                                     args.seed, extra_cfg)

    # Save results
    if args.output:
        import json
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {args.output}")

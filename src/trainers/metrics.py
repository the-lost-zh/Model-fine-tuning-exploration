"""Metrics collection utilities."""

import time
import torch
from .base_trainer import evaluate


def measure_inference_throughput(model, loader, device="cuda", warmup=10, repeat=50):
    """Measure inference throughput in images per second.

    Args:
        model: model to evaluate
        loader: DataLoader with batch_size to test
        device: torch device
        warmup: number of warmup batches
        repeat: number of timed batches

    Returns:
        imgs_per_sec: throughput measurement
    """
    model.eval()
    model = model.to(device)

    # Warmup
    iterator = iter(loader)
    for _ in range(warmup):
        try:
            x, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, _ = next(iterator)
        x = x.to(device)
        with torch.no_grad():
            _ = model(x)

    # Timed
    torch.cuda.synchronize()
    t0 = time.time()
    count = 0
    for _ in range(repeat):
        try:
            x, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, _ = next(iterator)
        x = x.to(device)
        with torch.no_grad():
            _ = model(x)
        count += x.size(0)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    return count / elapsed


def param_summary(model):
    """Return parameter summary dict.

    Returns:
        dict with: trainable_params, total_params, trainable_pct
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": 100 * trainable / total if total > 0 else 0,
    }

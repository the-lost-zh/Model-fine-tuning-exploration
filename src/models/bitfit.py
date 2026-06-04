"""BitFit: Bias-only Fine-tuning.

Freezes all model weights and trains only the bias terms (and the task-specific
classification head). Modifies ~0.08% of parameters while matching full
fine-tuning on small-to-medium datasets.

Reference: Zaken et al., "BitFit: Simple Parameter-efficient Fine-tuning
for Transformer-based Masked Language-models", ACL 2022.
"""

import torch.nn as nn


def apply_bitfit(model):
    """Apply BitFit: freeze everything except bias terms and classifier head.

    Args:
        model: ViT model from timm

    Returns:
        model with only bias terms and classifier head trainable
    """
    for name, param in model.named_parameters():
        if "bias" in name or "head" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model

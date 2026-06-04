"""LoRA: Low-Rank Adaptation.

Injects trainable low-rank decomposition matrices A and B into linear layers.
Applied to Q, K, V, and output projections in MSA blocks.
After training, A and B can be merged into the original weights for zero
inference overhead.

Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Linear layer wrapped with LoRA low-rank decomposition.

    h = W_0 x + (alpha / r) * B A x

    where A ∈ R^{r × in}, B ∈ R^{out × r}, r << min(in, out).
    """

    def __init__(self, linear: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.linear = linear
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        in_features = linear.in_features
        out_features = linear.out_features
        self.r = r
        self.scaling = alpha / r

        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        delta = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return self.linear(x) + delta

    def merge_weights(self):
        """Merge LoRA matrices into the original weight: W = W_0 + (alpha/r) * BA.

        After merging, this module behaves like a standard nn.Linear.
        """
        delta = (self.lora_B @ self.lora_A) * self.scaling
        self.linear.weight.data = self.linear.weight.data + delta
        self.lora_A.requires_grad = False
        self.lora_B.requires_grad = False

    def unmerge_weights(self):
        """Reverse the merge: W_0 = W - (alpha/r) * BA."""
        delta = (self.lora_B @ self.lora_A) * self.scaling
        self.linear.weight.data = self.linear.weight.data - delta


def _replace_linear_with_lora(module, r=8, alpha=16.0, target_prefixes=None):
    """Recursively replace nn.Linear layers with LoRALinear.

    Args:
        module: root module to search
        r: LoRA rank
        alpha: LoRA scaling factor
        target_prefixes: list of attribute name prefixes to target (e.g. ['qkv', 'attn.proj'])
                         If None, target all linear layers in attention blocks.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            # For ViT blocks, we target qkv projection and attention output projection
            if target_prefixes is None or any(name.startswith(p) for p in target_prefixes):
                setattr(module, name, LoRALinear(child, r=r, alpha=alpha))
        else:
            _replace_linear_with_lora(child, r=r, alpha=alpha, target_prefixes=target_prefixes)


def apply_lora(model, r=8, alpha=16.0):
    """Apply LoRA to a ViT model.

    Injects LoRA into all linear layers inside attention blocks:
    - QKV projection (qkv)
    - Attention output projection (proj)

    Args:
        model: ViT model from timm
        r: LoRA rank (default 8)
        alpha: LoRA scaling (default 16)

    Returns:
        model with LoRA applied
    """
    from .base_vit import freeze_all

    freeze_all(model)

    # Apply LoRA to each transformer block
    for block in model.blocks:
        _replace_linear_with_lora(
            block.attn, r=r, alpha=alpha,
            target_prefixes=["qkv", "proj"]
        )

    # Ensure classifier head is trainable
    if hasattr(model, "head"):
        for p in model.head.parameters():
            p.requires_grad = True

    return model

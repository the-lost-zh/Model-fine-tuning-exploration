"""AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition.

Inserts a lightweight parallel bottleneck branch (AdaptMLP) alongside the frozen
FFN/MLP in each ViT block. The original MLP is kept frozen while the parallel
branch learns task-specific features.

Output: x = MLP(LN(x)) + s * ReLU(LN(x) * W_down) * W_up + x

Reference: Chen et al., "AdaptFormer: Adapting Vision Transformers for
Scalable Visual Recognition", NeurIPS 2022.
"""

import torch
import torch.nn as nn


class AdaptMLP(nn.Module):
    """Parallel bottleneck adapter for ViT MLP blocks.

    Structure: Linear_down → ReLU → Linear_up, with learnable scale factor.
    Inserted in parallel with the frozen original MLP.
    """

    def __init__(self, d_model: int, d_hat: int = 64, scale: float = 0.1):
        super().__init__()
        self.down = nn.Linear(d_model, d_hat)
        self.up = nn.Linear(d_hat, d_model)
        self.scale = nn.Parameter(torch.tensor(scale))
        self.act = nn.ReLU()

        # Zero-initialize up-projection weights for stable training start
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        nn.init.zeros_(self.down.bias)

    def forward(self, x):
        return self.scale * self.up(self.act(self.down(x)))


class AdaptedMLPBlock(nn.Module):
    """Replacement for a ViT MLP block with parallel AdaptMLP branch.

    Original MLP (frozen) + AdaptMLP (trainable) in parallel.
    Both receive the same input (after LayerNorm).
    """

    def __init__(self, original_mlp: nn.Module, d_model: int,
                 d_hat: int = 64, scale: float = 0.1):
        super().__init__()
        self.original_mlp = original_mlp
        # Freeze the original MLP
        for p in self.original_mlp.parameters():
            p.requires_grad = False
        self.adapt_mlp = AdaptMLP(d_model, d_hat=d_hat, scale=scale)

    def forward(self, x):
        return self.original_mlp(x) + self.adapt_mlp(x)


def apply_adaptformer(model, d_hat: int = 64, scale: float = 0.1):
    """Apply AdaptFormer to a ViT model.

    Replaces the MLP block in each transformer layer with an AdaptedMLPBlock
    that adds a parallel AdaptMLP branch.

    Args:
        model: ViT model from timm
        d_hat: bottleneck middle dimension (default 64)
        scale: initial scale factor (default 0.1 for vision tasks)

    Returns:
        model with AdaptFormer applied
    """
    from .base_vit import freeze_all

    freeze_all(model)

    # ViT blocks from timm have: norm1, attn, norm2, mlp (or norm1, attn, drop_path1, norm2, mlp, drop_path2)
    # The MLP is accessible as block.mlp
    for block in model.blocks:
        d_model = block.mlp.fc1.in_features
        block.mlp = AdaptedMLPBlock(
            block.mlp, d_model, d_hat=d_hat, scale=scale
        )

    # Ensure classifier head is trainable
    if hasattr(model, "head"):
        for p in model.head.parameters():
            p.requires_grad = True

    return model

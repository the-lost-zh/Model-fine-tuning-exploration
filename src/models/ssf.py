"""SSF: Scaling & Shifting Your Features.

Inserts learnable scale (gamma) and shift (beta) parameters after each operation
(MSA, MLP, LayerNorm) in the ViT. After training, gamma and beta can be
re-parameterized (folded) into the preceding linear layer weights for zero
inference overhead.

Reference: Lian et al., "Scaling & Shifting Your Features: A New Baseline
for Efficient Model Tuning", NeurIPS 2022.
"""

import torch
import torch.nn as nn


class ScaleShift(nn.Module):
    """Channel-wise scale and shift: y = gamma * x + beta.

    gamma and beta have the same shape as the feature dimension.
    Can be re-parameterized into a preceding linear layer's weights.
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        return self.gamma * x + self.beta


def _insert_ssf_after_linear(module, attr_name, child):
    """Wrap a linear layer's output with ScaleShift."""
    num_features = child.out_features
    ssf_module = nn.Sequential(child, ScaleShift(num_features))
    setattr(module, attr_name, ssf_module)


def _insert_ssf_after_norm(module, attr_name, child):
    """Wrap a LayerNorm output with ScaleShift."""
    num_features = child.normalized_shape[0]
    ssf_module = nn.Sequential(child, ScaleShift(num_features))
    setattr(module, attr_name, ssf_module)


def _insert_ssf_after_op(module, attr_name, child):
    """Insert ScaleShift after a supported operation."""
    if isinstance(child, nn.Linear):
        _insert_ssf_after_linear(module, attr_name, child)
    elif isinstance(child, nn.LayerNorm):
        _insert_ssf_after_norm(module, attr_name, child)
    # Skip if already wrapped (e.g. Sequential from previous SSF injection)


def apply_ssf(model):
    """Apply SSF to a ViT model.

    Inserts ScaleShift modules after:
    - QKV projection, attention output projection (in MSA block)
    - fc1, fc2 (in MLP block)
    - norm1, norm2 (LayerNorm before MSA and MLP)

    Args:
        model: ViT model from timm

    Returns:
        model with SSF modules inserted
    """
    from .base_vit import freeze_all

    freeze_all(model)

    for block in model.blocks:
        # MSA block: insert SSF after qkv projection and attention proj
        if hasattr(block.attn, "qkv"):
            _insert_ssf_after_op(block.attn, "qkv", block.attn.qkv)
        if hasattr(block.attn, "proj"):
            _insert_ssf_after_op(block.attn, "proj", block.attn.proj)

        # MLP block: insert SSF after fc1 and fc2
        if hasattr(block.mlp, "fc1"):
            _insert_ssf_after_op(block.mlp, "fc1", block.mlp.fc1)
        if hasattr(block.mlp, "fc2"):
            _insert_ssf_after_op(block.mlp, "fc2", block.mlp.fc2)
        if hasattr(block.mlp, "act"):
            # Skip activation — SSF goes after linear layers only
            pass

        # LayerNorm: insert SSF after norm1 and norm2
        if hasattr(block, "norm1"):
            _insert_ssf_after_op(block, "norm1", block.norm1)
        if hasattr(block, "norm2"):
            _insert_ssf_after_op(block, "norm2", block.norm2)

    # Ensure classifier head is trainable
    if hasattr(model, "head"):
        for p in model.head.parameters():
            p.requires_grad = True

    return model


def reparameterize_ssf(model):
    """Fold SSF gamma/beta into preceding linear layer weights.

    For a linear layer W with optional bias b, followed by SSF(gamma, beta):
        y = gamma * (Wx + b) + beta = (gamma * W) x + (gamma * b + beta)

    After re-parameterization, ScaleShift modules can be removed.
    Currently only supports Linear -> ScaleShift (Sequential) pattern.

    Args:
        model: ViT model with SSF modules applied

    Returns:
        model with SSF parameters folded into preceding weights
    """
    for block in model.blocks:
        for parent in [block.attn, block.mlp, block]:
            for attr_name, child in list(parent.named_children()):
                if isinstance(child, nn.Sequential) and len(child) == 2:
                    linear_or_norm, ssf = child[0], child[1]
                    if not isinstance(ssf, ScaleShift):
                        continue

                    if isinstance(linear_or_norm, nn.Linear):
                        w = linear_or_norm.weight.data
                        b = linear_or_norm.bias.data if linear_or_norm.bias is not None else None
                        gamma = ssf.gamma.data
                        beta = ssf.beta.data

                        # W_new = gamma * W (broadcast over output dimension)
                        linear_or_norm.weight.data = gamma.unsqueeze(1) * w
                        if b is not None:
                            linear_or_norm.bias.data = gamma * b + beta
                        else:
                            linear_or_norm.bias = nn.Parameter(beta.clone())
                            linear_or_norm.bias.requires_grad = False

                        # Replace Sequential with just the linear layer
                        setattr(parent, attr_name, linear_or_norm)

    return model

"""SSF-Sparse: Gated SSF with L1 sparsity regularization.

Extends SSF by adding learnable gates that can disable channel-wise scale/shift
modulation when unnecessary. L1 sparsity loss pushes gates toward zero,
revealing which SSF positions are actually needed.

This is the innovation component for the course project.

Formulation:
    y = g ⊙ (γ ⊙ x + β) + (1 - g) ⊙ x

where g = sigmoid(gate_logit) ∈ [0,1]^d controls how much SSF modulation
passes through vs. being skipped (identity bypass).
"""

import torch
import torch.nn as nn


class SparseScaleShift(nn.Module):
    """Gated ScaleShift with sparsity regularization.

    Same as ScaleShift but with a learnable sigmoid gate per channel.
    When gate → 0, the channel bypasses SSF modulation entirely.
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.gate_logit = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        gate = torch.sigmoid(self.gate_logit)
        modulated = self.gamma * x + self.beta
        return gate * modulated + (1.0 - gate) * x

    def get_gate_values(self):
        """Return current gate activations (for analysis)."""
        return torch.sigmoid(self.gate_logit).detach()

    def sparsity_loss(self):
        """L1 loss on gate activations (sum of all gate values)."""
        return torch.sigmoid(self.gate_logit).sum()


def _insert_ssf_sparse_after_linear(module, attr_name, child):
    """Wrap a linear layer's output with SparseScaleShift."""
    num_features = child.out_features
    ssf_module = nn.Sequential(child, SparseScaleShift(num_features))
    setattr(module, attr_name, ssf_module)


def _insert_ssf_sparse_after_norm(module, attr_name, child):
    """Wrap a LayerNorm output with SparseScaleShift."""
    num_features = child.normalized_shape[0]
    ssf_module = nn.Sequential(child, SparseScaleShift(num_features))
    setattr(module, attr_name, ssf_module)


def _insert_ssf_sparse_after_op(module, attr_name, child):
    """Insert SparseScaleShift after a supported operation."""
    if isinstance(child, nn.Linear):
        _insert_ssf_sparse_after_linear(module, attr_name, child)
    elif isinstance(child, nn.LayerNorm):
        _insert_ssf_sparse_after_norm(module, attr_name, child)


def apply_ssf_sparse(model):
    """Apply SSF-Sparse to a ViT model.

    Same insertion points as standard SSF, but uses SparseScaleShift modules
    with learnable gates.

    Args:
        model: ViT model from timm

    Returns:
        model with SparseScaleShift modules inserted
    """
    from .base_vit import freeze_all

    freeze_all(model)

    for block in model.blocks:
        if hasattr(block.attn, "qkv"):
            _insert_ssf_sparse_after_op(block.attn, "qkv", block.attn.qkv)
        if hasattr(block.attn, "proj"):
            _insert_ssf_sparse_after_op(block.attn, "proj", block.attn.proj)

        if hasattr(block.mlp, "fc1"):
            _insert_ssf_sparse_after_op(block.mlp, "fc1", block.mlp.fc1)
        if hasattr(block.mlp, "fc2"):
            _insert_ssf_sparse_after_op(block.mlp, "fc2", block.mlp.fc2)

        if hasattr(block, "norm1"):
            _insert_ssf_sparse_after_op(block, "norm1", block.norm1)
        if hasattr(block, "norm2"):
            _insert_ssf_sparse_after_op(block, "norm2", block.norm2)

    if hasattr(model, "head"):
        for p in model.head.parameters():
            p.requires_grad = True

    return model


def collect_sparsity_loss(model):
    """Sum L1 gate loss across all SparseScaleShift modules."""
    total = 0.0
    for m in model.modules():
        if isinstance(m, SparseScaleShift):
            total += m.sparsity_loss()
    return total


def get_sparsity_stats(model):
    """Collect per-layer sparsity statistics for analysis.

    Returns:
        stats: list of dicts with keys: layer_idx, component (attn/mlp/norm),
               op_name, num_channels, mean_gate, frac_active (gate > 0.5)
    """
    stats = []
    for block_idx, block in enumerate(model.blocks):
        for parent_name, parent in [("attn", block.attn), ("mlp", block.mlp), ("norm", block)]:
            for attr_name, child in parent.named_children():
                if isinstance(child, nn.Sequential):
                    for sub in child:
                        if isinstance(sub, SparseScaleShift):
                            gates = sub.get_gate_values()
                            stats.append({
                                "layer": block_idx,
                                "component": parent_name,
                                "op": attr_name,
                                "num_channels": len(gates),
                                "mean_gate": gates.mean().item(),
                                "frac_active": (gates > 0.5).float().mean().item(),
                            })
    return stats


def prune_ssf_sparse(model, threshold=0.01):
    """Remove SparseScaleShift modules where mean gate < threshold.

    After pruning, the model size is reduced. Remaining modules can optionally
    be converted to standard ScaleShift (by removing the gate) and
    re-parameterized.

    Args:
        model: ViT model with SSF-Sparse applied
        threshold: gate values below this are considered inactive

    Returns:
        model with pruned SSF modules
    """
    for block in model.blocks:
        for parent in [block.attn, block.mlp, block]:
            for attr_name, child in list(parent.named_children()):
                if isinstance(child, nn.Sequential) and len(child) == 2:
                    _, ssf = child[0], child[1]
                    if isinstance(ssf, SparseScaleShift):
                        if ssf.get_gate_values().mean() < threshold:
                            # Replace Sequential with just the original op
                            setattr(parent, attr_name, child[0])

    return model

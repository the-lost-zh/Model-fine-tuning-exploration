"""Gate-LoRA: Gated LoRA + SSF hybrid module.

Combines LoRA (low-rank weight update) with SSF (channel modulation),
controlled by a shared learnable gate per output channel.

This is the second innovation component for the course project.

Formulation:
    base = W_0 x                          # frozen pre-trained weight
    lora = (alpha/r) * B @ A @ x          # LoRA low-rank update
    modulated = gamma * base + beta + lora # combined modulation
    gate = sigmoid(gate_logit)            # per-channel gate
    y = gate * modulated + (1 - gate) * base

When gate -> 0: identity bypass (original model)
When gate -> 1: full modulation (LoRA + SSF)
"""

import math
import torch
import torch.nn as nn


class GateLoRALinear(nn.Module):
    """Linear layer with gated LoRA + SSF.

    A single gate per output channel controls whether both LoRA and SSF
    modulation are applied. This reveals which channels actually benefit
    from additional trainable capacity.

    Args:
        linear: original nn.Linear layer (frozen)
        r: LoRA rank
        alpha: LoRA scaling factor
    """

    def __init__(self, linear: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.linear = linear
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        out_features = linear.out_features
        in_features = linear.in_features

        # LoRA parameters
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # SSF parameters (per-channel scale & shift)
        self.gamma = nn.Parameter(torch.ones(out_features))
        self.beta = nn.Parameter(torch.zeros(out_features))

        # Shared gate (controls both LoRA and SSF)
        self.gate_logit = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # Frozen base output
        base = self.linear(x)

        # LoRA low-rank update
        lora_delta = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

        # Combined modulation: SSF + LoRA
        modulated = self.gamma * base + self.beta + lora_delta

        # Gate controls how much modulation passes through
        gate = torch.sigmoid(self.gate_logit)
        return gate * modulated + (1.0 - gate) * base

    def get_gate_values(self):
        """Return current gate activations (for analysis)."""
        return torch.sigmoid(self.gate_logit).detach()

    def sparsity_loss(self):
        """L1 loss on gate activations."""
        return torch.sigmoid(self.gate_logit).sum()

    def merge_weights(self):
        """Merge LoRA into original weights (SSF can't be merged)."""
        delta = (self.lora_B @ self.lora_A) * self.scaling
        self.linear.weight.data = self.linear.weight.data + delta
        self.lora_A.requires_grad = False
        self.lora_B.requires_grad = False


def _replace_linear_with_gate_lora(module, r=8, alpha=16.0, target_prefixes=None):
    """Recursively replace nn.Linear layers with GateLoRALinear."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            if target_prefixes is None or any(name.startswith(p) for p in target_prefixes):
                setattr(module, name, GateLoRALinear(child, r=r, alpha=alpha))
        else:
            _replace_linear_with_gate_lora(child, r=r, alpha=alpha, target_prefixes=target_prefixes)


def apply_gate_lora(model, r=8, alpha=16.0):
    """Apply Gate-LoRA to a ViT model.

    Replaces QKV and attention output linear layers with GateLoRALinear modules.
    Freezes all base parameters, makes Gate-LoRA params + classifier head trainable.

    Args:
        model: ViT model from timm
        r: LoRA rank
        alpha: LoRA scaling

    Returns:
        model with Gate-LoRA applied
    """
    from .base_vit import freeze_all

    freeze_all(model)

    for block in model.blocks:
        _replace_linear_with_gate_lora(
            block.attn, r=r, alpha=alpha,
            target_prefixes=["qkv", "proj"]
        )

    if hasattr(model, "head"):
        for p in model.head.parameters():
            p.requires_grad = True

    return model


def collect_gate_lora_sparsity_loss(model):
    """Sum L1 gate loss across all GateLoRALinear modules."""
    total = 0.0
    for m in model.modules():
        if isinstance(m, GateLoRALinear):
            total += m.sparsity_loss()
    return total


def get_gate_lora_stats(model):
    """Collect per-layer gate statistics for Gate-LoRA modules.

    Returns:
        list of dicts with: block_idx, component, op_name,
        num_channels, mean_gate, frac_active (gate > 0.5)
    """
    stats = []
    for block_idx, block in enumerate(model.blocks):
        for attr_name, child in block.attn.named_children():
            if isinstance(child, GateLoRALinear):
                gates = child.get_gate_values()
                stats.append({
                    "block_idx": block_idx,
                    "component": "attn",
                    "op_name": attr_name[:4],  # 'qkv' or 'proj'
                    "num_channels": len(gates),
                    "mean_gate": gates.mean().item(),
                    "frac_active": (gates > 0.5).float().mean().item(),
                })
    return stats

"""ViT model loading and classification head wrapper.

Uses timm to load pre-trained ViT models and attaches a task-specific
classification head. The CLS token output is used for classification.

For users behind GFW, set environment variable before running:
    export HF_ENDPOINT=https://hf-mirror.com

Or pre-download weights and pass the local path via --pretrained_path.
"""

import os
import torch
import torch.nn as nn
import timm


def build_vit(model_name="vit_base_patch16_224", pretrained=True, num_classes=200,
              pretrained_path=None):
    """Load a pre-trained ViT from timm and replace the classifier head.

    Args:
        model_name: timm model identifier (e.g. 'vit_base_patch16_224')
        pretrained: load ImageNet pre-trained weights from HuggingFace
        num_classes: number of output classes for the downstream task
        pretrained_path: optional local path to pretrained weights (.pth)

    Returns:
        model: ViT model with replaced head
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )

    if pretrained_path is not None:
        state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {pretrained_path}")

    return model


def count_trainable_params(model):
    """Return (trainable_params, total_params)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def freeze_all(model):
    """Freeze all parameters in the model."""
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_classifier(model):
    """Unfreeze only the classifier head (for linear probing baseline)."""
    freeze_all(model)
    if hasattr(model, "head"):
        for p in model.head.parameters():
            p.requires_grad = True


def get_vit_blocks(model):
    """Return list of ViT transformer blocks."""
    return model.blocks

"""Unified datamodule for FGVC datasets.

Supports: CUB-200-2011, Oxford Flowers-102, Stanford Cars.

Each dataset is downloaded via torchvision and wrapped into standard
train/val/test DataLoader objects with configurable batch size and transforms.
"""

import os
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset, random_split

from .transforms import get_train_transforms, get_val_transforms
from .cub200 import CUB200


DATASET_CONFIGS = {
    "cub200": {
        "dir": "CUB_200_2011",
        "num_classes": 200,
        "class_name": CUB200,
    },
    "flowers102": {
        "dir": "Flowers102",
        "num_classes": 102,
        "class_name": datasets.Flowers102,
    },
    "stanford_cars": {
        "dir": "StanfordCars",
        "num_classes": 196,
        "class_name": datasets.StanfordCars,
    },
}


def build_dataloaders(dataset_name, data_root, batch_size=128, num_workers=8,
                      input_size=224, val_split=0.2):
    """Build train, validation, and test dataloaders for a dataset.

    Args:
        dataset_name: one of 'cub200', 'flowers102', 'stanford_cars'
        data_root: root directory where datasets are stored
        batch_size: batch size for all loaders
        num_workers: number of data loading workers
        input_size: image input size
        val_split: fraction of training data to use for validation

    Returns:
        train_loader, val_loader, test_loader, num_classes
    """
    config = DATASET_CONFIGS[dataset_name]
    data_dir = os.path.join(data_root, config["dir"])
    num_classes = config["num_classes"]

    train_transforms = get_train_transforms(input_size)
    val_transforms = get_val_transforms(input_size)

    # Some torchvision datasets have different APIs
    if dataset_name == "cub200":
        # CUB200: download=True, train=True for training set
        full_train = config["class_name"](
            root=data_root, train=True, download=True,
            transform=train_transforms,
        )
        test_set = config["class_name"](
            root=data_root, train=False, download=True,
            transform=val_transforms,
        )
    elif dataset_name == "flowers102":
        # Flowers102: split="train"/"val"/"test"
        full_train = config["class_name"](
            root=data_root, split="train", download=True,
            transform=train_transforms,
        )
        val_set = config["class_name"](
            root=data_root, split="val", download=True,
            transform=val_transforms,
        )
        test_set_raw = config["class_name"](
            root=data_root, split="test", download=True,
            transform=val_transforms,
        )
        # Flowers102 has its own val split; combine train+val for our use
        full_train = torch.utils.data.ConcatDataset([full_train, val_set])
        # Actually, let's keep it simple: use train as full, split ourselves
        # Re-download to avoid issues
        full_train = config["class_name"](
            root=data_root, split="train", download=True,
            transform=train_transforms,
        )
        test_set = config["class_name"](
            root=data_root, split="test", download=True,
            transform=val_transforms,
        )
    elif dataset_name == "stanford_cars":
        # StanfordCars: split="train"/"test"
        # download URL is broken, data must be pre-downloaded
        full_train = config["class_name"](
            root=data_root, split="train", download=False,
            transform=train_transforms,
        )
        test_set = config["class_name"](
            root=data_root, split="test", download=False,
            transform=val_transforms,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Split training into train/val
    total_train = len(full_train)
    val_size = int(total_train * val_split)
    train_size = total_train - val_size

    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(
        full_train, [train_size, val_size], generator=generator
    )

    # Validation subset uses val transforms
    val_subset.dataset.transform = val_transforms

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, num_classes


def get_subset_dataloader(dataset_name, data_root, fraction, batch_size=128,
                          num_workers=8, input_size=224, seed=42):
    """Build a dataloader using only a fraction of the training data.

    Used for sample efficiency experiments (Experiment 2).

    Args:
        fraction: fraction of training data to use (e.g. 0.1 for 10%)

    Returns:
        train_loader, val_loader, test_loader, num_classes
    """
    config = DATASET_CONFIGS[dataset_name]
    num_classes = config["num_classes"]

    train_transforms = get_train_transforms(input_size)
    val_transforms = get_val_transforms(input_size)

    data_dir = os.path.join(data_root, config["dir"])

    if dataset_name == "cub200":
        full_train = config["class_name"](
            root=data_root, train=True, download=True, transform=train_transforms,
        )
        test_set = config["class_name"](
            root=data_root, train=False, download=True, transform=val_transforms,
        )
    elif dataset_name == "flowers102":
        full_train = config["class_name"](
            root=data_root, split="train", download=True, transform=train_transforms,
        )
        test_set = config["class_name"](
            root=data_root, split="test", download=True, transform=val_transforms,
        )
    elif dataset_name == "stanford_cars":
        full_train = config["class_name"](
            root=data_root, split="train", download=False, transform=train_transforms,
        )
        test_set = config["class_name"](
            root=data_root, split="test", download=False, transform=val_transforms,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    total = len(full_train)
    subset_size = int(total * fraction)

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total, generator=generator)[:subset_size]
    subset = Subset(full_train, indices.tolist())

    # Split subset into train/val
    val_size = int(subset_size * 0.2)
    train_size = subset_size - val_size
    train_subset, val_subset = random_split(
        subset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, num_classes

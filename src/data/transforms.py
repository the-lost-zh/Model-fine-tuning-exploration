"""Data augmentation and transform pipelines for ViT fine-tuning.

Training: RandAugment + random resized crop + horizontal flip + normalization
Validation: resize + center crop + normalization (standard ImageNet eval)
"""

import torchvision.transforms as transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(input_size=224, auto_augment=True):
    """Training augmentation pipeline."""
    transforms_list = [
        transforms.RandomResizedCrop(input_size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]
    if auto_augment:
        transforms_list.append(transforms.RandAugment(num_ops=2, magnitude=9))
    transforms_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transforms.Compose(transforms_list)


def get_val_transforms(input_size=224):
    """Validation/Test transformation pipeline."""
    return transforms.Compose([
        transforms.Resize(int(input_size * 256 / 224)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

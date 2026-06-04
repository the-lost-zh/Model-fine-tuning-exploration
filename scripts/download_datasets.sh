#!/bin/bash
# Download FGVC datasets via torchvision.
# Datasets are downloaded on first use by the DataLoader code,
# but this script pre-downloads them for convenience.
#
# Usage: bash scripts/download_datasets.sh

set -e

DATA_ROOT="${1:-./data}"

echo "Downloading datasets to $DATA_ROOT ..."

python -c "
import torchvision.datasets as datasets
import os

# CUB-200-2011
print('Downloading CUB-200-2011...')
datasets.CUB200(root='$DATA_ROOT', train=True, download=True)
datasets.CUB200(root='$DATA_ROOT', train=False, download=True)

# Oxford Flowers-102
print('Downloading Flowers-102...')
datasets.Flowers102(root='$DATA_ROOT', split='train', download=True)
datasets.Flowers102(root='$DATA_ROOT', split='test', download=True)

# Stanford Cars
print('Downloading Stanford Cars...')
datasets.StanfordCars(root='$DATA_ROOT', split='train', download=True)
datasets.StanfordCars(root='$DATA_ROOT', split='test', download=True)

print('All datasets downloaded.')
"

echo "Done."

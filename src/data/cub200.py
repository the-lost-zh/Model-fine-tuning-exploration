"""Custom CUB-200-2011 dataset loader (not available in torchvision < 0.21)."""

import os
import tarfile
import requests
from PIL import Image
import torch
from torch.utils.data import Dataset


CUB_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"


class CUB200(Dataset):
    """CUB-200-2011 fine-grained bird classification dataset.

    Downloads and extracts the dataset if not present.
    Compatible with the torchvision-style API (root, train, transform).
    """

    def __init__(self, root, train=True, download=True, transform=None):
        self.root = root
        self.transform = transform

        self._data_dir = os.path.join(root, "CUB_200_2011")

        if download:
            self._download()

        images_path = os.path.join(self._data_dir, "images.txt")
        labels_path = os.path.join(self._data_dir, "image_class_labels.txt")
        split_path = os.path.join(self._data_dir, "train_test_split.txt")

        with open(images_path) as f:
            image_lines = f.read().strip().split("\n")
        with open(labels_path) as f:
            label_lines = f.read().strip().split("\n")
        with open(split_path) as f:
            split_lines = f.read().strip().split("\n")

        self.samples = []
        for img_line, lbl_line, spl_line in zip(image_lines, label_lines, split_lines):
            img_id, img_path = img_line.split(" ", 1)
            _, class_id = lbl_line.split(" ", 1)
            _, is_train = spl_line.split(" ", 1)

            is_training = int(is_train) == 1
            if train != is_training:
                continue

            full_path = os.path.join(self._data_dir, "images", img_path)
            self.samples.append((full_path, int(class_id) - 1))

    def _download(self):
        if os.path.exists(self._data_dir):
            return

        tgz_path = os.path.join(self.root, "CUB_200_2011.tgz")
        if not os.path.exists(tgz_path):
            os.makedirs(self.root, exist_ok=True)
            print(f"Downloading CUB-200-2011 from {CUB_URL}...")
            resp = requests.get(CUB_URL, stream=True)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with open(tgz_path, "wb") as f:
                downloaded = 0
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
            print("Download complete.")

        print("Extracting CUB-200-2011...")
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(path=self.root)
        print("Extraction complete.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

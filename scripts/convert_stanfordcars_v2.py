"""Convert StanfordCars parquet files to torchvision-compatible format (v2).

Fixes filename collision by using a global counter across all parquet files.
"""

import os
import numpy as np
import pandas as pd
import scipy.io as sio
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import shutil

import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
PARQUET_DIR = os.path.join(PROJECT_DIR, "data", "stanford_cars_parquet")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "data", "stanford_cars")


def convert_parquet(parquet_file, output_dir, global_idx, annotations):
    """Convert one parquet file and return updated global_idx."""
    pq_path = os.path.join(PARQUET_DIR, parquet_file)
    if not os.path.exists(pq_path):
        print(f"  WARNING: {parquet_file} not found")
        return global_idx

    print(f"  Processing {parquet_file}...")
    df = pd.read_parquet(pq_path)

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"    {parquet_file}"):
        img_bytes = row['image']['bytes']
        label = int(row['label'])

        fname = f"{global_idx:05d}.jpg"
        fpath = os.path.join(output_dir, fname)

        try:
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            img.save(fpath, "JPEG", quality=95)
        except Exception as e:
            print(f"    Error saving {fname}: {e}")
            continue

        annotations.append((fname, label + 1))
        global_idx += 1

    return global_idx


def main():
    # Clean previous output
    for d in ["cars_train", "cars_test"]:
        path = os.path.join(OUTPUT_DIR, d)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "devkit"), exist_ok=True)

    # Convert training data
    print("=== Converting training data ===")
    train_annos = []
    idx = 0
    idx = convert_parquet("train-00000-of-00002.parquet",
                          os.path.join(OUTPUT_DIR, "cars_train"), idx, train_annos)
    idx = convert_parquet("train-00001-of-00002.parquet",
                          os.path.join(OUTPUT_DIR, "cars_train"), idx, train_annos)
    print(f"  Total training images: {len(train_annos)}")

    # Convert test data
    print("=== Converting test data ===")
    test_annos = []
    idx = 0
    idx = convert_parquet("test-00000-of-00002.parquet",
                          os.path.join(OUTPUT_DIR, "cars_test"), idx, test_annos)
    idx = convert_parquet("test-00001-of-00002.parquet",
                          os.path.join(OUTPUT_DIR, "cars_test"), idx, test_annos)
    print(f"  Total test images: {len(test_annos)}")

    # Save annotations
    print("=== Saving annotations ===")
    dtype = np.dtype([('fname', 'O'), ('class', '<i4')])

    train_arr = np.array(train_annos, dtype=dtype)
    sio.savemat(os.path.join(OUTPUT_DIR, "devkit", "cars_train_annos.mat"),
                {"annotations": train_arr})

    test_arr = np.array(test_annos, dtype=dtype)
    sio.savemat(os.path.join(OUTPUT_DIR, "devkit", "cars_test_annos_withlabels.mat"),
                {"annotations": test_arr})

    # Create meta file
    class_names = np.array([f'Class_{i+1}' for i in range(196)], dtype=object)
    sio.savemat(os.path.join(OUTPUT_DIR, "devkit", "cars_meta.mat"),
                {'class_names': class_names})

    print(f"=== Conversion complete ===")
    print(f"Train images: {len(train_annos)}")
    print(f"Test images: {len(test_annos)}")


if __name__ == "__main__":
    main()

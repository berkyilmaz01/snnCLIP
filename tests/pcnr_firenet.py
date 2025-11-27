"""
Check PSNR of E2VID HQ images vs Ground Truth
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = "C:/Users/berky/PycharmProjects/ssn-spikeCLIP"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "spikeclip_snn"))

# Paths
HQ_DIR = os.path.join(PROJECT_ROOT, "datasets/HQ_FireNet")
GT_DIR = os.path.join(PROJECT_ROOT, "datasets/101_ObjectCategories/101_ObjectCategories")


def compute_psnr(img1, img2, max_val=1.0):
    """Compute PSNR between two images"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * np.log10(max_val / np.sqrt(mse))
    return psnr


def load_and_normalize(path, size=(180, 240)):
    """Load image and normalize to [0, 1]"""
    img = Image.open(path).convert("L")  # grayscale
    img = img.resize(size, Image.BILINEAR)
    return np.array(img).astype(np.float32) / 255.0


def main():
    hq_path = Path(HQ_DIR)
    gt_path = Path(GT_DIR)

    psnr_values = []
    class_psnrs = {}

    # Check a few classes
    classes_to_check = ["accordion", "airplanes", "anchor", "ant", "barrel"]

    print("Computing PSNR for E2VID HQ vs Ground Truth...\n")

    for class_name in classes_to_check:
        hq_class_dir = hq_path / class_name
        gt_class_dir = gt_path / class_name

        if not hq_class_dir.exists() or not gt_class_dir.exists():
            print(f"  Skipping {class_name} (not found)")
            continue

        # Get HQ images
        hq_images = sorted(hq_class_dir.glob("*.png"))[:10]  # first 10
        gt_images = sorted(gt_class_dir.glob("*.jpg"))[:10]

        class_psnr = []
        for i, (hq_img, gt_img) in enumerate(zip(hq_images, gt_images)):
            hq = load_and_normalize(hq_img)
            gt = load_and_normalize(gt_img, size=hq.shape[::-1])  # match HQ size

            psnr = compute_psnr(hq, gt)
            class_psnr.append(psnr)
            psnr_values.append(psnr)

        avg_class_psnr = np.mean(class_psnr)
        class_psnrs[class_name] = avg_class_psnr
        print(f"  {class_name}: {avg_class_psnr:.2f} dB (n={len(class_psnr)})")

    print(f"\n{'=' * 40}")
    print(f"Overall Average PSNR: {np.mean(psnr_values):.2f} dB")
    print(f"Min: {np.min(psnr_values):.2f} dB | Max: {np.max(psnr_values):.2f} dB")
    print(f"{'=' * 40}")

    # Compare with SNN baseline
    print(f"\nComparison:")
    print(f"  Your SNN (Stage 1):  ~9.8 dB")
    print(f"  E2VID HQ:            ~{np.mean(psnr_values):.1f} dB")
    print(f"  Quality gap:         ~{np.mean(psnr_values) - 9.8:.1f} dB")


if __name__ == "__main__":
    main()
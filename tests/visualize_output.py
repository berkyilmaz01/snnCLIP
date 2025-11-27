"""
SpikeCLIP Stage 1 Visualization
Uses same setup as train_spikeclip.py
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import numpy as np

from torch.utils.data import DataLoader, random_split
from data.ncaltech101_dataset import NCaltech101Dataset
from models.snn_model import SNNReconstruction

# Seed for reproducibility
# torch.manual_seed(42)
# np.random.seed(42)
# random.seed(42)

# Settings (same as training)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_BINS = 5
NUM_STEPS = 50
BETA = 0.95
BATCH_SIZE = 8

# Dataset Paths (same as training)
EVENT_PATH = "/datasets/N-Caltech101/Caltech101/Caltech101"
IMAGE_PATH = "/datasets/101_ObjectCategories/101_ObjectCategories"

print("=" * 60)
print("SpikeCLIP Stage 1 Visualization")
print("=" * 60)

# Load model
print("\n[1/4] Loading model...")
snn_model = SNNReconstruction(num_bins=NUM_BINS, beta=BETA).to(DEVICE)
checkpoint = torch.load("../spikeclip_snn/checkpoints/spikeclip_best.pth", map_location=DEVICE)
snn_model.load_state_dict(checkpoint['model_state_dict'])
snn_model.eval()
print(f"  ✓ Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
print(f"  ✓ Val loss: {checkpoint.get('val_loss', '?'):.4f}")
print(f"  ✓ Val PSNR: {checkpoint.get('val_psnr', '?'):.2f} dB")

# Load dataset (same as training)
print("\n[2/4] Loading dataset...")
dataset = NCaltech101Dataset(
    root_dir=EVENT_PATH,
    num_bins=NUM_BINS,
    image_dir=IMAGE_PATH
)
class_names = dataset.classes

# Use validation split (same as training)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
_, val_dataset = random_split(dataset, [train_size, val_size])

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,  # Shuffle to get random samples
    num_workers=0
)
print(f"  ✓ Loaded {len(val_dataset)} validation samples")

# Get a batch
print("\n[3/4] Generating outputs...")
voxels, images, labels = next(iter(val_loader))
voxels = voxels.to(DEVICE)
images = images.to(DEVICE)

with torch.no_grad():
    outputs = snn_model(voxels, num_steps=NUM_STEPS)

# Statistics
print("\n" + "=" * 60)
print("OUTPUT STATISTICS")
print("=" * 60)
print(f"  Output shape: {outputs.shape}")
print(f"  Output min:   {outputs.min().item():.4f}")
print(f"  Output max:   {outputs.max().item():.4f}")
print(f"  Output mean:  {outputs.mean().item():.4f}")
print(f"  Output std:   {outputs.std().item():.4f}")

print("\n" + "=" * 60)
print("MODE COLLAPSE CHECK")
print("=" * 60)
means = [outputs[i].mean().item() for i in range(BATCH_SIZE)]
stds = [outputs[i].std().item() for i in range(BATCH_SIZE)]
print(f"  Per-sample means: {[f'{m:.3f}' for m in means]}")
print(f"  Per-sample stds:  {[f'{s:.3f}' for s in stds]}")

mean_variance = np.var(means)
if mean_variance < 0.001:
    print("  ⚠️  WARNING: All outputs have very similar mean - POSSIBLE MODE COLLAPSE!")
else:
    print(f"  ✓ Outputs have different means (variance: {mean_variance:.6f})")

print("\n" + "=" * 60)
print("PER-SAMPLE PSNR")
print("=" * 60)
psnrs = []
for i in range(BATCH_SIZE):
    mse = F.mse_loss(outputs[i], images[i])
    if mse > 0:
        psnr = -10 * torch.log10(mse).item()
    else:
        psnr = float('inf')
    psnrs.append(psnr)
    print(f"  Sample {i} ({class_names[labels[i]]:20s}): PSNR = {psnr:.2f} dB, MSE = {mse.item():.4f}")

print(f"\n  Average PSNR: {np.mean(psnrs):.2f} dB")

# Visualization
print("\n[4/4] Creating visualization...")

fig, axes = plt.subplots(3, BATCH_SIZE, figsize=(BATCH_SIZE * 3, 9))

for i in range(BATCH_SIZE):
    # Row 0: Event voxel grid (sum across time bins)
    voxel_vis = voxels[i].cpu().sum(dim=0).numpy()
    axes[0, i].imshow(voxel_vis, cmap='hot')
    axes[0, i].set_title(f"Events\n{class_names[labels[i]]}", fontsize=8)
    axes[0, i].axis('off')

    # Row 1: Ground truth image
    axes[1, i].imshow(images[i, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[1, i].set_title("Ground Truth", fontsize=8)
    axes[1, i].axis('off')

    # Row 2: SNN output
    axes[2, i].imshow(outputs[i, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[2, i].set_title(f"SNN Output\nPSNR: {psnrs[i]:.1f} dB", fontsize=8)
    axes[2, i].axis('off')

# Add row labels
fig.text(0.02, 0.83, 'INPUT\n(Events)', ha='left', va='center', fontsize=10, fontweight='bold')
fig.text(0.02, 0.50, 'TARGET\n(GT Image)', ha='left', va='center', fontsize=10, fontweight='bold')
fig.text(0.02, 0.17, 'OUTPUT\n(SNN)', ha='left', va='center', fontsize=10, fontweight='bold')

plt.suptitle(f"SpikeCLIP Stage 1 Results - Epoch {checkpoint.get('epoch', '?')}", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.subplots_adjust(left=0.08)

# Save
save_path = "../spikeclip_snn/checkpoints/stage1_visualization.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\n  ✓ Saved to: {save_path}")

plt.show()

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
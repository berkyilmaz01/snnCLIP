"""
SpikeCLIP Stage 1 - Visualize ALL Classes
Shows one sample from each of the 101 classes
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader
from data.ncaltech101_dataset import NCaltech101Dataset
from models.snn_model import SNNReconstruction

# Settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_BINS = 5
NUM_STEPS = 50
BETA = 0.95

# Dataset Paths
EVENT_PATH = "/datasets/N-Caltech101/Caltech101/Caltech101"
IMAGE_PATH = "/datasets/101_ObjectCategories/101_ObjectCategories"

print("=" * 60)
print("SpikeCLIP Stage 1 - ALL CLASSES Visualization")
print("=" * 60)

# Load model
print("\n[1/3] Loading model...")
snn_model = SNNReconstruction(num_bins=NUM_BINS, beta=BETA).to(DEVICE)
checkpoint = torch.load("../spikeclip_snn/checkpoints/spikeclip_best.pth", map_location=DEVICE)
snn_model.load_state_dict(checkpoint['model_state_dict'])
snn_model.eval()
print(f"  ✓ Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

# Load dataset
print("\n[2/3] Loading dataset...")
dataset = NCaltech101Dataset(
    root_dir=EVENT_PATH,
    num_bins=NUM_BINS,
    image_dir=IMAGE_PATH
)
class_names = dataset.classes
num_classes = len(class_names)
print(f"  ✓ Found {num_classes} classes")

# Find one sample per class
print("\n[3/3] Finding one sample per class...")
class_samples = {}  # {class_idx: (voxel, image, label)}

for idx in range(len(dataset)):
    voxel, image, label = dataset[idx]
    if label not in class_samples:
        class_samples[label] = (voxel, image, label)
    if len(class_samples) == num_classes:
        break  # Found all classes

print(f"  ✓ Found samples for {len(class_samples)} classes")

# Generate outputs for all classes
print("\nGenerating outputs for all classes...")
all_voxels = []
all_images = []
all_labels = []
all_outputs = []
all_psnrs = []

for class_idx in sorted(class_samples.keys()):
    voxel, image, label = class_samples[class_idx]
    voxel = voxel.unsqueeze(0).to(DEVICE)
    image = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = snn_model(voxel, num_steps=NUM_STEPS)

    # Calculate PSNR
    mse = F.mse_loss(output, image)
    psnr = -10 * torch.log10(mse).item() if mse > 0 else float('inf')

    all_voxels.append(voxel.cpu())
    all_images.append(image.cpu())
    all_labels.append(label)
    all_outputs.append(output.cpu())
    all_psnrs.append(psnr)

print(f"  ✓ Generated {len(all_outputs)} outputs")
print(f"  ✓ Average PSNR: {np.mean(all_psnrs):.2f} dB")
print(f"  ✓ Min PSNR: {np.min(all_psnrs):.2f} dB")
print(f"  ✓ Max PSNR: {np.max(all_psnrs):.2f} dB")

# Create large visualization grid
# 101 classes → 11 rows x 10 columns (with one empty)
COLS = 10
ROWS = (num_classes + COLS - 1) // COLS  # Ceiling division

print(f"\nCreating {ROWS}x{COLS} visualization grid...")

# Create figure for Ground Truth vs Output comparison
fig, axes = plt.subplots(ROWS * 2, COLS, figsize=(COLS * 2.5, ROWS * 4))

for i, class_idx in enumerate(sorted(class_samples.keys())):
    row = (i // COLS) * 2  # Ground truth row
    col = i % COLS

    # Ground Truth (even rows)
    axes[row, col].imshow(all_images[i][0, 0].numpy(), cmap='gray', vmin=0, vmax=1)
    axes[row, col].set_title(f"{class_names[class_idx][:12]}", fontsize=6)
    axes[row, col].axis('off')

    # SNN Output (odd rows)
    axes[row + 1, col].imshow(all_outputs[i][0, 0].numpy(), cmap='gray', vmin=0, vmax=1)
    axes[row + 1, col].set_title(f"PSNR: {all_psnrs[i]:.1f}", fontsize=6)
    axes[row + 1, col].axis('off')

# Hide empty cells
for i in range(len(class_samples), ROWS * COLS):
    row = (i // COLS) * 2
    col = i % COLS
    if row < ROWS * 2:
        axes[row, col].axis('off')
        axes[row + 1, col].axis('off')

plt.suptitle(
    f"SpikeCLIP Stage 1 - All {num_classes} Classes\nTop: Ground Truth | Bottom: SNN Output (Epoch {checkpoint.get('epoch', '?')})",
    fontsize=12, fontweight='bold')
plt.tight_layout()

# Save
save_path = "../spikeclip_snn/checkpoints/stage1_all_classes.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\n  ✓ Saved to: {save_path}")

plt.show()

# Also save PSNR statistics
print("\n" + "=" * 60)
print("PSNR BY CLASS (sorted by PSNR)")
print("=" * 60)

# Sort by PSNR
sorted_indices = np.argsort(all_psnrs)[::-1]  # Descending

print("\nTOP 10 BEST:")
for i in sorted_indices[:10]:
    print(f"  {class_names[all_labels[i]]:25s}: {all_psnrs[i]:.2f} dB")

print("\nTOP 10 WORST:")
for i in sorted_indices[-10:]:
    print(f"  {class_names[all_labels[i]]:25s}: {all_psnrs[i]:.2f} dB")

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
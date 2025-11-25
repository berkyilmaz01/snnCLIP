import sys
sys.path.append('.')
from data.ncaltech101_dataset import NCaltech101Dataset

# print("=" * 50)
# print("Testing N-Caltech101 Dataset")
# print("=" * 50)
#
# # PATH
# dataset_path = "define path"
#
# print(f"\nLoading dataset from: {dataset_path}")
# dataset = NCaltech101Dataset(root_dir=dataset_path, num_bins=5)
#
# print(f"\nDataset size: {len(dataset)} samples")
# print(f"Number of classes: {len(dataset.classes)}")
#
# # Load first sample
# print("\nTesting loading first sample...")
# voxel, label = dataset[0]
# print(f"Voxel shape: {voxel.shape}")
# print(f"Voxel type: {voxel.dtype}")
# print(f"VClass label: {label} ({dataset.classes[label]})")

# Paths
dataset_path = ""
image_path = ""

# Test WITHOUT images first
print("\n--- Test 1: Event data only ---")
dataset = NCaltech101Dataset(root_dir=dataset_path, num_bins=5)
voxel, label = dataset[0]
print(f"    Voxel shape: {voxel.shape}")
print(f"    Class label: {label} ({dataset.classes[label]})")

# Test WITH images
print("\n--- Test 2: Event + Image pairs ---")
dataset_paired = NCaltech101Dataset(
    root_dir=dataset_path,
    num_bins=5,
    image_dir=image_path
)
voxel, image, label = dataset_paired[0]
print(f"    Voxel shape: {voxel.shape}")
print(f"    Image shape: {image.shape}")
print(f"    Image range: [{image.min():.3f}, {image.max():.3f}]")
print(f"    Class label: {label} ({dataset_paired.classes[label]})")
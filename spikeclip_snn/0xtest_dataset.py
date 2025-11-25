import sys
sys.path.append('.')
from data.ncaltech101_dataset import NCaltech101Dataset

print("=" * 50)
print("Testing N-Caltech101 Dataset")
print("=" * 50)

# PATH
dataset_path = "define path"

print(f"\nLoading dataset from: {dataset_path}")
dataset = NCaltech101Dataset(root_dir=dataset_path, num_bins=5)

print(f"\nDataset size: {len(dataset)} samples")
print(f"Number of classes: {len(dataset.classes)}")

# Load first sample
print("\nTesting loading first sample...")
voxel, label = dataset[0]
print(f"Voxel shape: {voxel.shape}")
print(f"Voxel type: {voxel.dtype}")
print(f"VClass label: {label} ({dataset.classes[label]})")
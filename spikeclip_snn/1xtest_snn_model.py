"""
Test script for SNN Reconstruction Model
"""

import torch
from models.snn_model import SNNReconstruction

print("=" * 50)
print("Testing SNN Reconstruction Model")
print("=" * 50)

# Create the model
print("\nCreating SNN model...")
model = SNNReconstruction(num_bins=5, beta=0.9)
print(f"    Model created successfully!")

# Create a dummy input (batch_size=2, channels=5, height=180, width=240)
print("\nCreating dummy input voxel grid...")
dummy_voxel = torch.randn(2, 5, 180, 240)
print(f"    Input shape: {dummy_voxel.shape}")

# Forward pass
print("\nRunning forward pass...")
output = model(dummy_voxel, num_steps=5)
print(f"    Output shape: {output.shape}")
print(f"    Output range: [{output.min():.3f}, {output.max():.3f}]")
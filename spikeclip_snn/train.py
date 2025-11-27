"""
Training script for SNN Image Reconstruction
"""
import os
os.makedirs('checkpoints', exist_ok=True)
print("Checkpoints directory ready\n")

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split
from data.ncaltech101_dataset import NCaltech101Dataset
from models.snn_model import SNNReconstruction

# Setting up the training config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
# If GPU is available use GPU, else CPU
# Process 8 samples at a time

BATCH_SIZE = 8
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
NUM_BINS = 5
BETA = 0.9
NUM_STEPS = 5

# Storage for plotting
train_losses = []
epoch_train_losses = []
epoch_val_losses = []

# Dataset Paths
EVENT_PATH = "C:/Users/berky/PycharmProjects/ssn-spikeCLIP/datasets/N-Caltech101/Caltech101/Caltech101"
IMAGE_PATH = "C:/Users/berky/PycharmProjects/ssn-spikeCLIP/datasets/101_ObjectCategories/101_ObjectCategories"

# Create a dataset and dataloader
print("\nLoading dataset...")
full_dataset = NCaltech101Dataset(
    root_dir=EVENT_PATH,
    num_bins=NUM_BINS,
    image_dir=IMAGE_PATH
)

# Split 80% train, 20% validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# Create DataLoader for batching
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"Created DataLoader with batch size {BATCH_SIZE}")

# Create the model
print("\nCreating SNN model...")
model = SNNReconstruction(num_bins=NUM_BINS, beta=BETA).to(device)
print("Model created and moved to device")

# Loss Function and Optimizer
# Mean Squared Error for image reconstruction
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
print(f"Using MSE loss and Adam optimizer (lr={LEARNING_RATE})")


def validate(model, val_loader, criterion, device, num_steps):
    """
    Evaluate model on validation set

    Args:
        model: The SNN model to evaluate
        val_loader: DataLoader for validation data
        criterion: Loss function (MSE)
        device: CPU or CUDA
        num_steps: Number of timesteps for SNN


    Returns:
        avg_val_loss: Average loss over validation set
    """
    # Set to evaluation mode
    model.eval()
    val_loss = 0.0

    # No gradients needed for validation
    with torch.no_grad():
        for voxels, images, labels in val_loader:
            # Move to device
            voxels = voxels.to(device)
            images = images.to(device)

            # Forward pass
            outputs = model(voxels, num_steps=num_steps)
            loss = criterion(outputs, images)

            # Accumulate loss
            val_loss += loss.item()

    # Calculate average
    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss

# Training loop
print("\n" + "="*50)
print("Starting Training...")
print("="*50)

# Track the best validation loss
best_val_loss = float("inf")
best_epoch = 0

# Loop through all epochs
# Train the model
for epoch in range(NUM_EPOCHS):
    model.train()  # Set model to training mode
    epoch_loss = 0.0

    for batch_idx, (voxels, images, labels) in enumerate(train_loader):
        # Move data to device
        voxels = voxels.to(device)
        images = images.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(voxels, num_steps=NUM_STEPS)
        loss = criterion(outputs, images)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track loss
        epoch_loss += loss.item()
        train_losses.append(loss.item())

        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    # Print epoch summary
    avg_train_loss = epoch_loss / len(train_loader)
    epoch_train_losses.append(avg_train_loss)

    # Validate after each epoch
    avg_val_loss = validate(model, val_loader, criterion, device, NUM_STEPS)
    epoch_val_losses.append(avg_val_loss)

    # Check if this is the best model so far
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch + 1
        # Save best model
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"New best model! Val loss: {best_val_loss:.4f}")

    # Save regular checkpoint
    checkpoint_path = f"checkpoints/snn_epoch_{epoch + 1}.pth"

    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
    }, checkpoint_path)

    print(f"Checkpoint saved: {checkpoint_path}")
    print(f"\n>>> Epoch [{epoch + 1}/{NUM_EPOCHS}] Complete")
    print(f"    Training Loss:   {avg_train_loss:.4f}")
    print(f"    Validation Loss: {avg_val_loss:.4f}\n")

# Save the model
print("\n" + "="*50)
print("Training Complete. Saving final model...")
torch.save(model.state_dict(), "snn_reconstruction_final.pth")
print(f"Final model saved as 'snn_reconstruction_final.pth' (epoch {NUM_EPOCHS})")
print(f"Best model saved as 'best_model.pth' (epoch {best_epoch}, val_loss: {best_val_loss:.4f})")
print("="*50)

# Plot batch-wise loss
# Plot training curves
print("\nGenerating training plots...")
plt.figure(figsize=(15, 5))

# Plot
plt.subplot(1, 3, 1)
plt.plot(train_losses)
plt.title('Training Loss per Batch')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.grid(True)

# Epoch losses - Both train and val
plt.subplot(1, 3, 2)
plt.plot(range(1, NUM_EPOCHS + 1), epoch_train_losses, marker='o', label='Train')
plt.plot(range(1, NUM_EPOCHS + 1), epoch_val_losses, marker='s', label='Validation')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.legend()
plt.grid(True)

# Overfitting gap
plt.subplot(1, 3, 3)
gap = [t - v for t, v in zip(epoch_train_losses, epoch_val_losses)]
plt.plot(range(1, NUM_EPOCHS + 1), gap, marker='d', color='red')
plt.title('Train-Val Gap')
plt.xlabel('Epoch')
plt.ylabel('Loss Difference')
plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
plt.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png')
print("Training curves saved as 'training_curves.png'")


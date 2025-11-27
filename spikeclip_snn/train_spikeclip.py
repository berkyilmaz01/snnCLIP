"""
SpikeCLIP Traning Script with InfoNCE Loss
Uses CLIP-based supervision

# I added train.py file to observe the differences
# between dual pair images with supervised learning
# It uses both N-Caltech101 and Caltech101 datasets

"""
import os
# Check the checkpoints dir, if not create
os.makedirs("checkpoints", exist_ok=True)

# Imports
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import clip
import random

from torch.utils.data import DataLoader, random_split
from data.ncaltech101_dataset import NCaltech101Dataset
from models.snn_model import SNNReconstruction
from utils.clip_utils import info_nce_loss, prepare_for_clip, calculate_psnr

# seed
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# For tracking the best model
best_val_loss = float('inf')

# Check for device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
NUM_EPOCHS = 50
NUM_BINS = 5
BETA = 0.95
NUM_STEPS = 50
TEMPERATURE = 0.07

# Storage for plotting
train_losses = []
epoch_train_losses = []
epoch_val_losses = []
psnr_values = []

# Dataset Paths
EVENT_PATH = "C:/Users/berky/PycharmProjects/ssn-spikeCLIP/datasets/N-Caltech101/Caltech101/Caltech101"
IMAGE_PATH = "C:/Users/berky/PycharmProjects/ssn-spikeCLIP/datasets/101_ObjectCategories/101_ObjectCategories"

# First we need to load the CLIP model
# However, it is not trained during training
# Therefore, freeze the CLIP weights
# .required_grad for freezing the weights
print("\nLoading CLIP model...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model = clip_model.float()
clip_model.eval()
for param in clip_model.parameters():
    param.requires_grad = False
print("CLIP model loaded and frozen")



# Dataset Loading
print("\nLoading N-Caltech101 dataset!")
dataset = NCaltech101Dataset(
    root_dir=EVENT_PATH,
    num_bins=NUM_BINS,
    image_dir=IMAGE_PATH
)

# Get class names for text prompts
class_names = dataset.classes
num_classes = len(class_names)
print(f"Found {len(dataset)} samples across {num_classes} classes")

# Train/Val Split (80/20)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

# DataLoaders
# *Note, num_workers = 0 because it has been run on Windows
# To avoid errors
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

# Initialize SNN Model
print("\nInitializing SNN model!")
snn_model = SNNReconstruction(
    num_bins=NUM_BINS,
    beta=BETA
).to(device)

# Get all the parameters in the model
total_params = sum(p.numel() for p in snn_model.parameters())
trainable_params = sum(p.numel() for p in snn_model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Optimizer (only SNN parameters, CLIP is frozen)
optimizer = optim.Adam(snn_model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# If setup is complete:
print("\nSetup complete!")

# Training Loop
print("\n" + "="*50)
print("Starting Training!")
print("="*50)

for epoch in range(NUM_EPOCHS):
    # Training phase
    snn_model.train()
    epoch_loss = 0.0

    for batch_idx, (voxel, images, labels) in enumerate(train_loader):
        # Move to device
        # Images is for PSNR monitoring only
        voxel = voxel.to(device)
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass thorugh SNN
        outputs = snn_model(voxel, num_steps = NUM_STEPS)

        # Prepeare for CLIP
        outputs_clip = prepare_for_clip(outputs, device)

        # Create text prompts from class labels
        templates = [
            "a photo of a {}",
            "a cropped photo of the {}",
            "a clear photo of a {}",
            "an image of a {}",
        ]
        texts = [random.choice(templates).format(class_names[label.item()]) for label in labels]
        text_tokens = clip.tokenize(texts).to(device)

        # Get CLIP features, no gradients
        image_features = clip_model.encode_image(outputs_clip)
        with torch.no_grad():
            text_features = clip_model.encode_text(text_tokens)

        # Calculate the InfoNCE loss
        loss = info_nce_loss(image_features, text_features, TEMPERATURE)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # DEBUG: Check if gradients exist
        # if batch_idx == 0 and epoch == 0:
        #     for name, param in snn_model.named_parameters():
        #         if param.grad is not None:
        #             print(f"{name}: grad_mean = {param.grad.mean().item():.6f}")
        #         else:
        #             print(f"{name}: NO GRADIENT!")


        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(snn_model.parameters(), max_norm=1.0)
        optimizer.step()

        # Track Loss
        epoch_loss += loss.item()
        train_losses.append(loss.item())

        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")


    # Average training loss, going to next phase
    # Get the current loss values
    avg_train_loss = epoch_loss / len(train_loader)
    epoch_train_losses.append(avg_train_loss)

    # Validation
    # Turn the model for evaluation mode
    snn_model.eval()
    val_loss = 0.0
    val_psnr = 0.0

    with torch.no_grad():
        for voxels, images, labels in val_loader:

            # Move to device
            voxels = voxels.to(device)
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = snn_model(voxels, num_steps=NUM_STEPS)

            # Prepare for CLIP
            outputs_clip = prepare_for_clip(outputs, device)

            # Create text prompts
            templates = [
                "a photo of a {}",
                "a cropped photo of the {}",
                "a clear photo of a {}",
                "an image of a {}",
            ]
            texts = [random.choice(templates).format(class_names[l.item()]) for l in labels]
            text_tokens = clip.tokenize(texts).to(device)

            # Get CLIP features
            image_features = clip_model.encode_image(outputs_clip)
            text_features = clip_model.encode_text(text_tokens)

            # Calculate loss
            loss = info_nce_loss(image_features, text_features, TEMPERATURE)
            val_loss += loss.item()

            # Calculate PSNR (using ground truth images)
            psnr = calculate_psnr(outputs, images)
            val_psnr += psnr.item()

    # Average validation metrics
    avg_val_loss = val_loss / len(val_loader)
    avg_val_psnr = val_psnr / len(val_loader)

    epoch_val_losses.append(avg_val_loss)
    psnr_values.append(avg_val_psnr)

    # Print epoch summary
    print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] Summary:")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Val Loss:   {avg_val_loss:.4f}")
    print(f"  Val PSNR:   {avg_val_psnr:.2f} dB")
    print("="*50)

    # We should save the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': snn_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_psnr': avg_val_psnr
        }, "checkpoints/spikeclip_best.pth")
        print(f"New best model saved!")

    print("=" * 50)
    scheduler.step()

# Save final model
torch.save(snn_model.state_dict(), "checkpoints/spikeclip_final.pth")
print(f"\nTraining complete!")
print(f"Best validation loss: {best_val_loss:.4f}")
print("Best model saved to: checkpoints/spikeclip_best.pth")

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Training Loss (per batch)
axes[0].plot(train_losses)
axes[0].set_title("Training Loss (per batch)")
axes[0].set_xlabel("Batch")
axes[0].set_ylabel("InfoNCE Loss")

# Plot 2: Train vs Val Loss (per epoch)
axes[1].plot(epoch_train_losses, label="Train")
axes[1].plot(epoch_val_losses, label="Val")
axes[1].set_title("Train vs Val Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("InfoNCE Loss")
axes[1].legend()

# Plot 3: PSNR over epochs
axes[2].plot(psnr_values, color='green')
axes[2].set_title("Validation PSNR")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("PSNR (dB)")

plt.tight_layout()
plt.savefig("checkpoints/spikeclip_training_curves.png")
plt.show()
print("Training curves saved to checkpoints/spikeclip_training_curves.png")

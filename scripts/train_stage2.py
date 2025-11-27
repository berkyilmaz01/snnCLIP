"""
SpikeCLIP Stage 2: Prompt Learning
Following SpikeCLIP paper Section 3.2
https://arxiv.org/abs/2501.04477

Learns HQ/LQ prompts to distinguish high-quality from low-quality images
Uses frozen SNN from Stage 1 to generate LQ images
"""

# Imports
import os
import sys

PROJECT_ROOT = "C:/Users/berky/PycharmProjects/ssn-spikeCLIP"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "spikeclip_snn"))


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import clip
from torch.utils.data import DataLoader, random_split

# Local imports
from configs.stage2_config import (
    EVENT_PATH, IMAGE_PATH, STAGE1_CHECKPOINT, STAGE2_CHECKPOINT_DIR,
    NUM_BINS, BETA, NUM_STEPS, N_CTX,
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, TEMPERATURE, DEVICE, DEGRADATION_LEVEL,HQ_DATASET_DIR
)
from data.ncaltech101_dataset import NCaltech101Dataset
from data.hq_lq_dataset_e2vid import HQLQDatasetE2VID
from models.snn_model import SNNReconstruction
from models.prompt_learner import PromptCLIP
from losses.prompt_loss import PromptLoss, compute_accuracy

# Seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Create checkpoint directory
os.makedirs(STAGE2_CHECKPOINT_DIR, exist_ok=True)

def main():
    print("=" * 60)
    print("SpikeCLIP Stage 2: Prompt Learning")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # Load the CLIP model
    print("\n[1/6] Loading CLIP model")
    clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
    clip_model = clip_model.float()
    clip_model.eval()
    print("CLIP loaded")

    # Load Stage 1 SNN Model
    print("\n[2/6] Loading Stage 1 SNN model...")
    snn_model = SNNReconstruction(num_bins=NUM_BINS, beta=BETA).to(DEVICE)

    checkpoint = torch.load(STAGE1_CHECKPOINT, map_location=DEVICE)
    snn_model.load_state_dict(checkpoint['model_state_dict'])
    snn_model.eval()

    # Freeze SNN
    for param in snn_model.parameters():
        param.requires_grad = False

    print(f"    Loaded from epoch {checkpoint.get('epoch', '?')}")
    print(f"    Stage 1 Val Loss: {checkpoint.get('val_loss', '?'):.4f}")
    print(f"    Stage 1 Val PSNR: {checkpoint.get('val_psnr', '?'):.2f} dB")

    # Load Event Dataset (for LQ generation)
    print("\n[3/6] Loading event dataset...")
    event_dataset = NCaltech101Dataset(
        root_dir=EVENT_PATH,
        num_bins=NUM_BINS,
        image_dir=IMAGE_PATH
    )
    print(f"   Loaded {len(event_dataset)} samples")

    # Create HQ/LQ Dataset
    print("\n[4/6] Creating HQ/LQ dataset...")
    hqlq_dataset = HQLQDatasetE2VID(
        hq_dir=HQ_DATASET_DIR,
        snn_model=snn_model,
        event_dataset=event_dataset,
        device=DEVICE,
        num_steps=NUM_STEPS,
        samples_per_class=10
    )

    # Train/Val split
    train_size = int(0.8 * len(hqlq_dataset))
    val_size = len(hqlq_dataset) - train_size
    train_dataset, val_dataset = random_split(hqlq_dataset, [train_size, val_size])

    print(f"    Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # DataLoaders
    # Data already on GPU for train_loder
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    print(f"    Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Initialize PromptCLIP Model
    print("\n[5/6] Initializing PromptCLIP...")
    model = PromptCLIP(clip_model, n_ctx=N_CTX).to(DEVICE)

    # Loss and optimizer (only prompt parameters!)
    criterion = PromptLoss(label_smoothing=0.1)
    optimizer = optim.Adam(
        model.prompt_learner.parameters(),
        lr=LEARNING_RATE
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # Training Loop
    print("\n[6/6] Starting training...")
    print("=" * 60)

    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    val_accs = []

    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward
            logits = model(images)
            loss = criterion(logits, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            epoch_loss += loss.item()
            preds = logits.argmax(dim=1)
            epoch_correct += (preds == labels).sum().item()
            epoch_total += labels.size(0)

            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Batch [{batch_idx + 1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} Acc: {epoch_correct / epoch_total:.4f}")

        avg_train_loss = epoch_loss / len(train_loader)
        train_acc = epoch_correct / epoch_total
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                logits = model(images)
                loss = criterion(logits, labels)

                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)

        # Print epoch summary
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print("=" * 60)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'prompt_learner_state_dict': model.prompt_learner.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_acc': val_acc,
                'n_ctx': N_CTX
            }, os.path.join(STAGE2_CHECKPOINT_DIR, "prompt_best.pth"))
            print(f"  ✓ New best model saved! (Val Acc: {val_acc:.4f})")

        scheduler.step()

    # Save the final model
    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'prompt_learner_state_dict': model.prompt_learner.state_dict(),
        'n_ctx': N_CTX
    }, os.path.join(STAGE2_CHECKPOINT_DIR, "prompt_final.pth"))

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"  Best Val Accuracy: {best_val_acc:.4f}")
    print(f"  Checkpoints saved to: {STAGE2_CHECKPOINT_DIR}")
    print("=" * 60)

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss
        axes[0].plot(train_losses, label='Train')
        axes[0].plot(val_losses, label='Val')
        axes[0].set_title('Stage 2: Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Cross-Entropy Loss')
        axes[0].legend()

        # Accuracy
        axes[1].plot(val_accs, color='green')
        axes[1].set_title('Stage 2: Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')

        plt.tight_layout()
        plt.savefig(os.path.join(STAGE2_CHECKPOINT_DIR, "stage2_training_curves.png"))
        plt.show()
        print(f"\n    Training curves saved!")
    except Exception as e:
        print(f"\n    Could not plot: {e}")


if __name__ == "__main__":
    main()























































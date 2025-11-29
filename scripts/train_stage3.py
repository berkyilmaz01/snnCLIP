"""
SpikeCLIP Stage 3: Quality-Guided SNN Fine-tuning
Uses learned HQ/LQ prompts to improve reconstruction quality
"""
import os
import sys

PROJECT_ROOT = "C:/Users/berky/PycharmProjects/ssn-spikeCLIP"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "spikeclip_snn"))

# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import clip
from torch.utils.data import DataLoader, random_split

from configs.stage3_config import (
    EVENT_PATH, IMAGE_PATH, STAGE1_CHECKPOINT, STAGE2_CHECKPOINT, STAGE3_CHECKPOINT_DIR,
    NUM_BINS, BETA, NUM_STEPS, N_CTX,
    LAMBDA_QUALITY, LAMBDA_RECON, LAMBDA_TV, LAMBDA_INFONCE, USE_EVENT_WEIGHTING,
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, TEMPERATURE, DEVICE
)
from data.ncaltech101_dataset import NCaltech101Dataset
from models.snn_model import SNNReconstruction
from models.prompt_learner import PromptCLIP
from losses.quality_loss import CombinedStage3Loss

# Seed
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Create checkpoint directory
os.makedirs(STAGE3_CHECKPOINT_DIR, exist_ok=True)

def compute_psnr(output, target):
    """Compute PSNR between output and target."""
    mse = F.mse_loss(output, target)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(1.0 / mse).item()

def main():
    print("=" * 60)
    print("SpikeCLIP Stage 3: Quality-Guided SNN Fine-tuning")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Loss weights: quality={LAMBDA_QUALITY}, recon={LAMBDA_RECON}, tv={LAMBDA_TV}, infonce={LAMBDA_INFONCE}")
    print(f"Event-aware weighting: {USE_EVENT_WEIGHTING}")

    # Load CLIP model
    print("\n[1/6] Loading CLIP model...")
    clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
    clip_model = clip_model.float().eval()

    # Freeze CLIP
    for param in clip_model.parameters():
        param.requires_grad = False
    print("     CLIP loaded and frozen")

    # Load stage 1 SSN, will be fine-tuned
    print("\n[2/6] Loading Stage 1 SNN model...")
    snn_model = SNNReconstruction(num_bins=NUM_BINS, beta=BETA).to(DEVICE)

    stage1_ckpt = torch.load(STAGE1_CHECKPOINT, map_location=DEVICE)
    snn_model.load_state_dict(stage1_ckpt['model_state_dict'])

    print(f"  Loaded from epoch {stage1_ckpt.get('epoch', '?')}")
    print(f"  Stage 1 PSNR: {stage1_ckpt.get('val_psnr', '?'):.2f} dB")

    # Load Stage 2 Prompts (frozen)
    print("\n[3/6] Loading Stage 2 learned prompts...")
    prompt_model = PromptCLIP(clip_model, n_ctx=N_CTX).to(DEVICE)

    stage2_ckpt = torch.load(STAGE2_CHECKPOINT, map_location=DEVICE)
    prompt_model.load_state_dict(stage2_ckpt['model_state_dict'])
    prompt_model.eval()

    # Freeze prompt model
    for param in prompt_model.parameters():
        param.requires_grad = False

    # Extract HQ/LQ text features
    with torch.no_grad():
        text_features = prompt_model.get_prompt_features()  # (2, 512)
        text_features_lq = text_features[0]  # (512,)
        text_features_hq = text_features[1]  # (512,)

    print(f"  Stage 2 Val Acc: {stage2_ckpt.get('val_acc', '?'):.4f}")
    print(f"  T_lq and T_hq extracted")

    # Load Dataset
    print("\n[4/6] Loading dataset...")
    dataset = NCaltech101Dataset(
        root_dir=EVENT_PATH,
        num_bins=NUM_BINS,
        image_dir=IMAGE_PATH
    )

    # Train/Val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"    Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # Setup Training
    # Loss
    criterion = CombinedStage3Loss(
        lambda_quality=LAMBDA_QUALITY,
        lambda_recon=LAMBDA_RECON,
        lambda_tv=LAMBDA_TV,
        lambda_infonce=LAMBDA_INFONCE,
        temperature=TEMPERATURE,
        use_event_weighting=USE_EVENT_WEIGHTING
    )

    # Optimizer (only SNN parameters)
    optimizer = optim.Adam(snn_model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # CLIP preprocessing
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(DEVICE)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(DEVICE)

    print(f"  Optimizer: Adam (lr={LEARNING_RATE})")
    print(f"  Training SNN parameters only")

    # Training Loop
    print("\n[6/6] Starting training...")
    print("=" * 60)

    best_val_psnr = 0.0
    train_losses = []
    val_psnrs = []

    for epoch in range(NUM_EPOCHS):
        # Training
        snn_model.train()
        epoch_loss = 0.0
        epoch_psnr = 0.0

        for batch_idx, (voxels, gt_images, labels) in enumerate(train_loader):
            voxels = voxels.to(DEVICE)
            gt_images = gt_images.to(DEVICE)

            # Forward through SNN
            snn_output = snn_model(voxels, num_steps=NUM_STEPS)

            # Prepare for CLIP (resize to 224x224, convert to RGB)
            snn_224 = F.interpolate(snn_output, size=(224, 224), mode='bilinear', align_corners=False)
            snn_rgb = snn_224.repeat(1, 3, 1, 1)
            snn_norm = (snn_rgb - clip_mean) / clip_std

            # Get CLIP image features
            # NOTE: No no_grad() here - we need gradients to flow back to SNN through quality loss
            # CLIP encoder is frozen, so only SNN will receive gradients
            image_features = clip_model.encode_image(snn_norm)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Compute loss
            loss, loss_dict = criterion(
                snn_output, gt_images,
                image_features.float(),
                text_features_hq,
                text_features_lq,
                event_voxel=voxels  # Pass voxels for event-aware weighting
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            epoch_loss += loss.item()
            with torch.no_grad():
                batch_psnr = compute_psnr(snn_output, gt_images)
                epoch_psnr += batch_psnr

            # Print progress
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} (Q:{loss_dict['quality']:.3f} R:{loss_dict['recon']:.3f} "
                      f"TV:{loss_dict.get('tv', 0):.3f}) PSNR: {batch_psnr:.2f}")

        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_psnr = epoch_psnr / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        snn_model.eval()
        val_psnr = 0.0
        val_quality_scores = []

        with torch.no_grad():
            for voxels, gt_images, labels in val_loader:
                voxels = voxels.to(DEVICE)
                gt_images = gt_images.to(DEVICE)

                snn_output = snn_model(voxels, num_steps=NUM_STEPS)
                val_psnr += compute_psnr(snn_output, gt_images)

                # Check quality score improvement
                snn_224 = F.interpolate(snn_output, size=(224, 224), mode='bilinear', align_corners=False)
                snn_rgb = snn_224.repeat(1, 3, 1, 1)
                snn_norm = (snn_rgb - clip_mean) / clip_std

                image_features = clip_model.encode_image(snn_norm)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                sim_hq = (image_features.float() @ text_features_hq).mean().item()
                val_quality_scores.append(sim_hq)


        avg_val_psnr = val_psnr / len(val_loader)
        avg_quality_score = np.mean(val_quality_scores)
        val_psnrs.append(avg_val_psnr)

        # Print epoch summary
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train PSNR: {avg_train_psnr:.2f} dB")
        print(f"  Val PSNR: {avg_val_psnr:.2f} dB | Quality Score: {avg_quality_score:.4f}")
        print("=" * 60)

        # Save best model
        if avg_val_psnr > best_val_psnr:
            best_val_psnr = avg_val_psnr
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': snn_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_psnr': avg_val_psnr,
                'quality_score': avg_quality_score
            }, os.path.join(STAGE3_CHECKPOINT_DIR, "snn_stage3_best.pth"))
            print(f"  ✓ New best model saved! (Val PSNR: {avg_val_psnr:.2f} dB)")

        scheduler.step()

    # Save final model
    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': snn_model.state_dict(),
        'val_psnr': avg_val_psnr
    }, os.path.join(STAGE3_CHECKPOINT_DIR, "snn_stage3_final.pth"))

    print("\n" + "=" * 60)
    print("Stage 3 Training Complete!")
    print(f"  Best Val PSNR: {best_val_psnr:.2f} dB")
    print(f"  Stage 1 PSNR was: {stage1_ckpt.get('val_psnr', 0):.2f} dB")
    print(f"  Improvement: {best_val_psnr - stage1_ckpt.get('val_psnr', 0):.2f} dB")
    print("=" * 60)

    # Plot
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(train_losses)
        axes[0].set_title('Stage 3: Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')

        axes[1].plot(val_psnrs, color='green')
        axes[1].axhline(y=stage1_ckpt.get('val_psnr', 0), color='red', linestyle='--', label='Stage 1 PSNR')
        axes[1].set_title('Stage 3: Validation PSNR')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('PSNR (dB)')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(STAGE3_CHECKPOINT_DIR, "stage3_training_curves.png"))
        plt.show()
    except Exception as e:
        print(f"Could not plot: {e}")


if __name__ == "__main__":
    main()





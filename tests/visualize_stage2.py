"""
Visualize Stage 2 Results:
- HQ vs LQ image samples
- Learned prompt features
- Model predictions
"""

import os
import sys

PROJECT_ROOT = "C:/Users/berky/PycharmProjects/ssn-spikeCLIP"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "spikeclip_snn"))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import clip
from PIL import Image
from pathlib import Path

from configs.stage2_config import (
    EVENT_PATH, IMAGE_PATH, STAGE1_CHECKPOINT, STAGE2_CHECKPOINT_DIR,
    NUM_BINS, BETA, NUM_STEPS, N_CTX, DEVICE
)
from data.ncaltech101_dataset import NCaltech101Dataset
from models.snn_model import SNNReconstruction
from models.prompt_learner import PromptCLIP


def main():
    print("=" * 60)
    print("Stage 2 Visualization: HQ vs LQ Images")
    print("=" * 60)

    # Load Models
    print("\n[1/4] Loading models...")

    # CLIP
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
    clip_model = clip_model.float().eval()

    # SNN (Stage 1)
    snn_model = SNNReconstruction(num_bins=NUM_BINS, beta=BETA).to(DEVICE)
    snn_checkpoint = torch.load(STAGE1_CHECKPOINT, map_location=DEVICE)
    snn_model.load_state_dict(snn_checkpoint['model_state_dict'])
    snn_model.eval()

    # PromptCLIP (Stage 2)
    prompt_model = PromptCLIP(clip_model, n_ctx=N_CTX).to(DEVICE)
    stage2_checkpoint = torch.load(
        os.path.join(STAGE2_CHECKPOINT_DIR, "prompt_best.pth"),
        map_location=DEVICE
    )
    prompt_model.load_state_dict(stage2_checkpoint['model_state_dict'])
    prompt_model.eval()

    print(f"  Stage 2 Val Acc: {stage2_checkpoint.get('val_acc', 'N/A'):.4f}")

    # Load Data
    print("\n[2/4] Loading data...")

    event_dataset = NCaltech101Dataset(
        root_dir=EVENT_PATH,
        num_bins=NUM_BINS,
        image_dir=IMAGE_PATH
    )

    # CLIP normalization
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1).to(DEVICE)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1).to(DEVICE)

    # Generate Samples
    print("\n[3/4] Generating samples...")

    num_samples = 8
    indices = np.random.choice(len(event_dataset), num_samples, replace=False)

    hq_images = []
    lq_images = []
    hq_preds = []
    lq_preds = []
    class_names = []

    with torch.no_grad():
        for idx in indices:
            voxel, gt_image, label = event_dataset[idx]
            class_names.append(event_dataset.classes[label])

            # === HQ Image (Ground Truth) ===
            hq_img = gt_image.unsqueeze(0).to(DEVICE)  # (1, 1, H, W)
            hq_img = F.interpolate(hq_img, size=(224, 224), mode='bilinear', align_corners=False)
            hq_img = hq_img.repeat(1, 3, 1, 1)  # Grayscale to RGB
            hq_img_norm = (hq_img - clip_mean.view(1, 3, 1, 1)) / clip_std.view(1, 3, 1, 1)

            # HQ prediction
            hq_logits = prompt_model(hq_img_norm)
            hq_pred = hq_logits.softmax(dim=1)[0]  # [LQ_prob, HQ_prob]
            hq_preds.append(hq_pred.cpu().numpy())
            hq_images.append(hq_img[0].cpu())

            # === LQ Image (SNN Output) ===
            voxel = voxel.unsqueeze(0).to(DEVICE)
            lq_img = snn_model(voxel, num_steps=NUM_STEPS)  # (1, 1, H, W)
            lq_img = F.interpolate(lq_img, size=(224, 224), mode='bilinear', align_corners=False)
            lq_img = lq_img.repeat(1, 3, 1, 1)  # Grayscale to RGB
            lq_img_norm = (lq_img - clip_mean.view(1, 3, 1, 1)) / clip_std.view(1, 3, 1, 1)

            # LQ prediction
            lq_logits = prompt_model(lq_img_norm)
            lq_pred = lq_logits.softmax(dim=1)[0]  # [LQ_prob, HQ_prob]
            lq_preds.append(lq_pred.cpu().numpy())
            lq_images.append(lq_img[0].cpu())

    # Visualize
    print("\n[4/4] Creating visualization...")

    fig, axes = plt.subplots(4, num_samples, figsize=(2.5 * num_samples, 10))

    for i in range(num_samples):
        # Row 1: HQ Images
        hq_display = hq_images[i][0].numpy()  # Take first channel
        axes[0, i].imshow(hq_display, cmap='gray')
        axes[0, i].set_title(f'{class_names[i][:10]}', fontsize=8)
        axes[0, i].axis('off')

        # Row 2: HQ Predictions
        axes[1, i].bar(['LQ', 'HQ'], hq_preds[i], color=['red', 'green'])
        axes[1, i].set_ylim(0, 1)
        axes[1, i].set_title(f'HQ:{hq_preds[i][1]:.2f}', fontsize=8)

        # Row 3: LQ Images
        lq_display = lq_images[i][0].numpy()  # Take first channel
        axes[2, i].imshow(lq_display, cmap='gray')
        axes[2, i].set_title('SNN Output', fontsize=8)
        axes[2, i].axis('off')

        # Row 4: LQ Predictions
        axes[3, i].bar(['LQ', 'HQ'], lq_preds[i], color=['red', 'green'])
        axes[3, i].set_ylim(0, 1)
        axes[3, i].set_title(f'LQ:{lq_preds[i][0]:.2f}', fontsize=8)

    # Row labels
    axes[0, 0].set_ylabel('HQ Image\n(Ground Truth)', fontsize=10)
    axes[1, 0].set_ylabel('HQ Prediction', fontsize=10)
    axes[2, 0].set_ylabel('LQ Image\n(SNN Output)', fontsize=10)
    axes[3, 0].set_ylabel('LQ Prediction', fontsize=10)

    plt.suptitle('Stage 2: HQ vs LQ Classification\n'
                 'Green = High Quality, Red = Low Quality', fontsize=12)
    plt.tight_layout()

    # Save
    save_path = os.path.join(STAGE2_CHECKPOINT_DIR, "stage2_hq_lq_visualization.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n✓ Saved to: {save_path}")

    # Print Prompt Analysis
    print("\n" + "=" * 60)
    print("Learned Prompt Analysis:")
    print("=" * 60)

    with torch.no_grad():
        # Get text features
        text_features = prompt_model.get_prompt_features()  # (2, 512)

        # Cosine similarity between LQ and HQ prompts
        similarity = (text_features[0] @ text_features[1]).item()

        print(f"  LQ prompt feature norm: {text_features[0].norm():.4f}")
        print(f"  HQ prompt feature norm: {text_features[1].norm():.4f}")
        print(f"  LQ-HQ cosine similarity: {similarity:.4f}")
        print(f"  (Lower = more distinct prompts)")

        # Learned ctx tokens
        ctx = prompt_model.prompt_learner.ctx
        print(f"\n  Learned [V1-V4] token stats:")
        print(f"    Shape: {ctx.shape}")
        print(f"    Mean: {ctx.mean():.4f}")
        print(f"    Std: {ctx.std():.4f}")
        print(f"    Min: {ctx.min():.4f}")
        print(f"    Max: {ctx.max():.4f}")


if __name__ == "__main__":
    main()
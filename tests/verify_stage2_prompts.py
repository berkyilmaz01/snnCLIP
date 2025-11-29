"""
Verify Stage 2 Learned Prompts
Comprehensive verification that Stage 2 learned meaningful HQ/LQ prompts

Checks:
1. Validation accuracy (should be > 0.7 ideally)
2. HQ images get high HQ probability
3. LQ images get high LQ probability
4. Prompt features are distinct (low cosine similarity)
5. Visual inspection of predictions
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
from torch.utils.data import DataLoader, random_split

from configs.stage2_config import (
    EVENT_PATH, IMAGE_PATH, STAGE1_CHECKPOINT, STAGE2_CHECKPOINT_DIR,
    NUM_BINS, BETA, NUM_STEPS, N_CTX, DEVICE
)
from data.ncaltech101_dataset import NCaltech101Dataset
from models.snn_model import SNNReconstruction
from models.prompt_learner import PromptCLIP


def verify_stage2_prompts():
    """Comprehensive verification of Stage 2 prompts."""
    print("=" * 70)
    print("Stage 2 Prompt Verification")
    print("=" * 70)

    # Load Models
    print("\n[1/5] Loading models...")
    
    clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
    clip_model = clip_model.float().eval()
    
    # SNN (Stage 1)
    snn_model = SNNReconstruction(num_bins=NUM_BINS, beta=BETA).to(DEVICE)
    snn_checkpoint = torch.load(STAGE1_CHECKPOINT, map_location=DEVICE)
    snn_model.load_state_dict(snn_checkpoint['model_state_dict'])
    snn_model.eval()
    
    # PromptCLIP (Stage 2)
    prompt_model = PromptCLIP(clip_model, n_ctx=N_CTX).to(DEVICE)
    stage2_checkpoint_path = os.path.join(STAGE2_CHECKPOINT_DIR, "prompt_best.pth")
    
    if not os.path.exists(stage2_checkpoint_path):
        print(f"  ERROR: Stage 2 checkpoint not found at {stage2_checkpoint_path}")
        return False
    
    stage2_checkpoint = torch.load(stage2_checkpoint_path, map_location=DEVICE)
    prompt_model.load_state_dict(stage2_checkpoint['model_state_dict'])
    prompt_model.eval()
    
    val_acc = stage2_checkpoint.get('val_acc', 0.0)
    print(f"  ✓ Stage 2 checkpoint loaded")
    print(f"  ✓ Validation Accuracy: {val_acc:.4f}")

    # Check 1: Validation Accuracy
    print("\n[2/5] Check 1: Validation Accuracy")
    print("-" * 70)
    if val_acc >= 0.8:
        print(f"  ✓ EXCELLENT: {val_acc:.2%} - Prompts learned very well!")
    elif val_acc >= 0.7:
        print(f"  ✓ GOOD: {val_acc:.2%} - Prompts learned reasonably well")
    elif val_acc >= 0.6:
        print(f"  ⚠ WARNING: {val_acc:.2%} - Prompts may not be distinct enough")
    else:
        print(f"  ✗ POOR: {val_acc:.2%} - Prompts did not learn properly!")
        print("     Consider:")
        print("     - Training Stage 2 for more epochs")
        print("     - Checking HQ/LQ dataset quality")
        print("     - Adjusting learning rate")

    # Check 2: Prompt Feature Analysis
    print("\n[3/5] Check 2: Prompt Feature Analysis")
    print("-" * 70)
    
    with torch.no_grad():
        text_features = prompt_model.get_prompt_features()  # (2, 512)
        text_features_lq = text_features[0]
        text_features_hq = text_features[1]
        
        # Cosine similarity
        similarity = (text_features_lq @ text_features_hq).item()
        
        # Norms
        norm_lq = text_features_lq.norm().item()
        norm_hq = text_features_hq.norm().item()
        
        print(f"  LQ prompt norm: {norm_lq:.4f}")
        print(f"  HQ prompt norm: {norm_hq:.4f}")
        print(f"  LQ-HQ cosine similarity: {similarity:.4f}")
        
        if similarity < 0.5:
            print(f"  ✓ GOOD: Prompts are distinct (similarity < 0.5)")
        elif similarity < 0.7:
            print(f"  ⚠ WARNING: Prompts are somewhat similar (similarity {similarity:.2f})")
        else:
            print(f"  ✗ POOR: Prompts are too similar (similarity {similarity:.2f})")
            print("     They may not distinguish HQ from LQ well")

    # Check 3: Test on Real Images
    print("\n[4/5] Check 3: Testing on Real Images")
    print("-" * 70)
    
    # Load dataset
    event_dataset = NCaltech101Dataset(
        root_dir=EVENT_PATH,
        num_bins=NUM_BINS,
        image_dir=IMAGE_PATH
    )
    
    # Use validation split
    train_size = int(0.8 * len(event_dataset))
    val_size = len(event_dataset) - train_size
    _, val_dataset = random_split(event_dataset, [train_size, val_size])
    
    # Sample test images
    num_test = 50
    test_indices = np.random.choice(len(val_dataset), min(num_test, len(val_dataset)), replace=False)
    
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(DEVICE)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(DEVICE)
    
    hq_correct = 0
    lq_correct = 0
    hq_hq_probs = []
    lq_lq_probs = []
    
    with torch.no_grad():
        for idx in test_indices:
            voxel, gt_image, label = val_dataset[idx]
            
            # === HQ Image (Ground Truth) ===
            hq_img = gt_image.unsqueeze(0).to(DEVICE)
            hq_img = F.interpolate(hq_img, size=(224, 224), mode='bilinear', align_corners=False)
            hq_img = hq_img.repeat(1, 3, 1, 1)
            hq_img_norm = (hq_img - clip_mean) / clip_std
            
            hq_logits = prompt_model(hq_img_norm)
            hq_pred = hq_logits.softmax(dim=1)[0]  # [LQ_prob, HQ_prob]
            hq_hq_probs.append(hq_pred[1].item())
            if hq_pred[1] > 0.5:  # Predicted as HQ
                hq_correct += 1
            
            # === LQ Image (SNN Output) ===
            voxel_batch = voxel.unsqueeze(0).to(DEVICE)
            lq_img = snn_model(voxel_batch, num_steps=NUM_STEPS)
            lq_img = F.interpolate(lq_img, size=(224, 224), mode='bilinear', align_corners=False)
            lq_img = lq_img.repeat(1, 3, 1, 1)
            lq_img_norm = (lq_img - clip_mean) / clip_std
            
            lq_logits = prompt_model(lq_img_norm)
            lq_pred = lq_logits.softmax(dim=1)[0]  # [LQ_prob, HQ_prob]
            lq_lq_probs.append(lq_pred[0].item())
            if lq_pred[0] > 0.5:  # Predicted as LQ
                lq_correct += 1
    
    hq_acc = hq_correct / len(test_indices)
    lq_acc = lq_correct / len(test_indices)
    avg_hq_prob = np.mean(hq_hq_probs)
    avg_lq_prob = np.mean(lq_lq_probs)
    
    print(f"  Tested on {len(test_indices)} image pairs")
    print(f"  HQ images → HQ prediction: {hq_acc:.2%} (avg HQ prob: {avg_hq_prob:.3f})")
    print(f"  LQ images → LQ prediction: {lq_acc:.2%} (avg LQ prob: {avg_lq_prob:.3f})")
    
    if hq_acc >= 0.7 and lq_acc >= 0.7:
        print(f"  ✓ GOOD: Both HQ and LQ are correctly identified")
    elif hq_acc >= 0.6 and lq_acc >= 0.6:
        print(f"  ⚠ WARNING: Some misclassifications, but mostly correct")
    else:
        print(f"  ✗ POOR: Many misclassifications")
        print("     Prompts may not be working correctly")

    # Check 4: Visual Inspection
    print("\n[5/5] Check 4: Visual Inspection")
    print("-" * 70)
    
    # Sample a few images for visualization
    num_vis = 6
    vis_indices = test_indices[:num_vis]
    
    fig, axes = plt.subplots(4, num_vis, figsize=(2.5 * num_vis, 10))
    if num_vis == 1:
        axes = axes.reshape(-1, 1)
    
    hq_images_vis = []
    lq_images_vis = []
    hq_preds_vis = []
    lq_preds_vis = []
    
    with torch.no_grad():
        for i, idx in enumerate(vis_indices):
            voxel, gt_image, label = val_dataset[idx]
            
            # HQ
            hq_img = gt_image.unsqueeze(0).to(DEVICE)
            hq_img = F.interpolate(hq_img, size=(224, 224), mode='bilinear', align_corners=False)
            hq_img = hq_img.repeat(1, 3, 1, 1)
            hq_img_norm = (hq_img - clip_mean) / clip_std
            hq_logits = prompt_model(hq_img_norm)
            hq_pred = hq_logits.softmax(dim=1)[0]
            
            # LQ
            voxel_batch = voxel.unsqueeze(0).to(DEVICE)
            lq_img = snn_model(voxel_batch, num_steps=NUM_STEPS)
            lq_img = F.interpolate(lq_img, size=(224, 224), mode='bilinear', align_corners=False)
            lq_img = lq_img.repeat(1, 3, 1, 1)
            lq_img_norm = (lq_img - clip_mean) / clip_std
            lq_logits = prompt_model(lq_img_norm)
            lq_pred = lq_logits.softmax(dim=1)[0]
            
            hq_images_vis.append(hq_img[0, 0].cpu().numpy())
            lq_images_vis.append(lq_img[0, 0].cpu().numpy())
            hq_preds_vis.append(hq_pred.cpu().numpy())
            lq_preds_vis.append(lq_pred.cpu().numpy())
    
    # Plot
    for i in range(num_vis):
        # Row 1: HQ Images
        axes[0, i].imshow(hq_images_vis[i], cmap='gray')
        axes[0, i].set_title('HQ (GT)', fontsize=8)
        axes[0, i].axis('off')
        
        # Row 2: HQ Predictions
        axes[1, i].bar(['LQ', 'HQ'], hq_preds_vis[i], color=['red', 'green'])
        axes[1, i].set_ylim(0, 1)
        axes[1, i].set_title(f'HQ prob: {hq_preds_vis[i][1]:.2f}', fontsize=8)
        
        # Row 3: LQ Images
        axes[2, i].imshow(lq_images_vis[i], cmap='gray')
        axes[2, i].set_title('LQ (SNN)', fontsize=8)
        axes[2, i].axis('off')
        
        # Row 4: LQ Predictions
        axes[3, i].bar(['LQ', 'HQ'], lq_preds_vis[i], color=['red', 'green'])
        axes[3, i].set_ylim(0, 1)
        axes[3, i].set_title(f'LQ prob: {lq_preds_vis[i][0]:.2f}', fontsize=8)
    
    axes[0, 0].set_ylabel('HQ Image', fontsize=10)
    axes[1, 0].set_ylabel('HQ Prediction', fontsize=10)
    axes[2, 0].set_ylabel('LQ Image', fontsize=10)
    axes[3, 0].set_ylabel('LQ Prediction', fontsize=10)
    
    plt.suptitle(f'Stage 2 Prompt Verification\nVal Acc: {val_acc:.2%} | '
                 f'HQ→HQ: {hq_acc:.2%} | LQ→LQ: {lq_acc:.2%}', fontsize=12)
    plt.tight_layout()
    
    save_path = os.path.join(STAGE2_CHECKPOINT_DIR, "stage2_verification.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Visualization saved to: {save_path}")
    plt.show()

    # Final Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    all_good = True
    if val_acc < 0.7:
        print("  ✗ Validation accuracy too low")
        all_good = False
    if similarity > 0.7:
        print("  ✗ Prompts too similar")
        all_good = False
    if hq_acc < 0.7 or lq_acc < 0.7:
        print("  ✗ Test accuracy too low")
        all_good = False
    
    if all_good:
        print("  ✓ ALL CHECKS PASSED: Stage 2 prompts are meaningful!")
        print("  ✓ Safe to proceed to Stage 3")
    else:
        print("  ⚠ SOME CHECKS FAILED: Review Stage 2 training")
        print("  ⚠ Consider retraining Stage 2 before Stage 3")
    
    print("=" * 70)
    
    return all_good


if __name__ == "__main__":
    verify_stage2_prompts()


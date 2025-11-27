"""
CLIP utilities for SpikeCLIP training
Contains InfoNCE loss and CLIP preprocessing helpers
"""

import torch
import torch.nn.functional as F

# CLIP normalization values
# Using the same mean and standard deviation values
# used when CLIP was trained
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


# infoNCE loss
def info_nce_loss(image_features, text_features, temperature = 0.07):
    """
        https://arxiv.org/pdf/2501.04477
    Contrastive loss between image and text embeddings.
    :param image_features: CLIP embeddings of reconstructed images [B, 512]
    :param text_features: CLIP embeddings of text prompts [B, 512]
    :param temperature: Scaling factor (default 0.07)
    :return: Scalar loss value
    """

    # Normalize the features to unit vectors
    image_features = F.normalize(image_features, dim = 1)
    text_features = F.normalize(text_features, dim = -1)

    # Compute the similarity matrix [B,B]
    logits = torch.matmul(image_features, text_features.T) / temperature

    # Labels, diagnoal elements are positive pairs
    labels = torch.arange(len(logits), device = image_features.device)

    # Cross entropy for both directions
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)

    # Average for both directions
    loss = (loss_i2t + loss_t2i) / 2

    return loss

def prepare_for_clip(images, device):
    """
    Prepare reconstructed images for CLIP encoding
    Resize to 224x224 (CLIP input size)
    Convert grayscale to RGB
    Normalize with CLIP stats

    :param images: Tensor of shape [B, 1, H, W]
    :param device:torch device (cuda/cpu)
    :return:
    """
    # Grayscale
    images = images.repeat(1, 3, 1, 1)

    # Resize to CLIP input size
    images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

    # Clamp to [0,1] just in case
    images = torch.clamp(images, 0, 1)

    # Normalize with CLIP mean/std
    mean = CLIP_MEAN.to(device)
    std = CLIP_STD.to(device)
    images = (images - mean) / std

    return images


def calculate_psnr(pred, target):
    """
    Calculate Peak Signal-to-Noise Ratio

    :param pred: Predicted image tenso
    :param target: Ground truth image tensor
    :return: PSNR value in dB
    """
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse)
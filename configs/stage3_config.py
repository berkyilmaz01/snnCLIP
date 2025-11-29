"""
Stage 3 Configuration: Quality-Guided SNN Fine-tuning
Following SpikeCLIP paper Section 3.3
https://arxiv.org/abs/2501.04477
"""

# Import libraries
import torch

# Paths
EVENT_PATH = "C:/Users/berky/PycharmProjects/ssn-spikeCLIP/datasets/N-Caltech101/Caltech101/Caltech101"
IMAGE_PATH = "C:/Users/berky/PycharmProjects/ssn-spikeCLIP/datasets/101_ObjectCategories/101_ObjectCategories"
STAGE1_CHECKPOINT = "C:/Users/berky/PycharmProjects/ssn-spikeCLIP/spikeclip_snn/checkpoints/spikeclip_best.pth"
STAGE2_CHECKPOINT = "C:/Users/berky/PycharmProjects/ssn-spikeCLIP/spikeclip_snn/checkpoints/stage2/prompt_best.pth"
STAGE3_CHECKPOINT_DIR = "C:/Users/berky/PycharmProjects/ssn-spikeCLIP/spikeclip_snn/checkpoints/stage3"

# Model Settings
NUM_BINS = 5
BETA = 0.95
NUM_STEPS = 50
N_CTX = 4

# Stage 3 Loss Settings
# Weight for quality loss (push toward HQ) - reduced to prevent over-smoothing
# Weight for reconstruction loss (preserve content) - increased to preserve details
# Weight for TV loss (smoothness, reduce artifacts)
# Weight for original InfoNCE loss (semantic alignment)
LAMBDA_QUALITY = 0.3
LAMBDA_RECON = 1.5
LAMBDA_TV = 0.4
LAMBDA_INFONCE = 0.5

# Event-aware weighting: weight reconstruction loss by event density
# Higher weight where events occur (edges/motion), lower in static regions
USE_EVENT_WEIGHTING = True

# Training Settings
# Smaller batch for fine-tuning
# Lower LR for fine-tuning
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
NUM_EPOCHS = 30
TEMPERATURE = 0.07

# Check device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
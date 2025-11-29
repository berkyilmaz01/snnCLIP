"""
Stage 2 Configuration: Prompt Learning
Following SpikeCLIP paper Section 3.2
https://arxiv.org/abs/2501.04477
"""
import torch

EVENT_PATH = "C:/Users/berky/PycharmProjects/ssn-spikeCLIP/datasets/N-Caltech101/Caltech101/Caltech101"
IMAGE_PATH = "C:/Users/berky/PycharmProjects/ssn-spikeCLIP/datasets/101_ObjectCategories/101_ObjectCategories"
STAGE1_CHECKPOINT = "C:/Users/berky/PycharmProjects/ssn-spikeCLIP/spikeclip_snn/checkpoints/spikeclip_best.pth"
STAGE2_CHECKPOINT_DIR = "C:/Users/berky/PycharmProjects/ssn-spikeCLIP/spikeclip_snn/checkpoints/stage2"


# Model Settings
NUM_BINS = 5
BETA = 0.95
NUM_STEPS = 50

# We need to define the number of learnable
# context tokens and CLIP embedding dimensions
# Also, token poisiton to identify where to
# put the class token
N_CTX = 4
CTX_DIM = 512
CLASS_TOKEN_POSITION = "end"

# Training settings
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
TEMPERATURE = 0.07

# DEGRADATION_LEVEL
DEGRADATION_LEVEL = "match_snn"

# Check device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HQ_DATASET_DIR = "C:/Users/berky/PycharmProjects/ssn-spikeCLIP/datasets/HQ_Dataset"

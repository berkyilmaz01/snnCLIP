"""
Generate HQ Dataset using FireNet
Run this once to create HQ reconstructions from N-Caltech101 events

"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# PATHS
PROJECT_ROOT = "C:/Users/berky/PycharmProjects/ssn-spikeCLIP"
FIRENET_PATH = os.path.join(PROJECT_ROOT, "firenet")
CHECKPOINT_PATH = os.path.join(FIRENET_PATH, "pretrained", "E2VID_lightweight.pth.tar")
NCALTECH_ROOT = os.path.join(PROJECT_ROOT, "datasets/N-Caltech101/Caltech101/Caltech101")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "datasets/HQ_FireNet")

# Add paths
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, FIRENET_PATH)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "spikeclip_snn"))

def load_firenet(checkpoint_path, device):
    """Load E2VIDRecurrent model"""
    from model.model import E2VIDRecurrent

    # E2VIDRecurrent uses config dict
    config = {
        'num_bins': 5,
        'base_num_channels': 32,
        'num_encoders': 3,  # checkpoint has 3 encoders
        'num_residual_blocks': 2,
        'skip_type': 'sum',
        'norm': 'BN',  # checkpoint uses batch norm
        'use_upsample_conv': False,  # uses transposed conv
        'recurrent_block_type': 'convlstm'
    }

    model = E2VIDRecurrent(config)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print(f"    E2VIDRecurrent loaded from {checkpoint_path}")
    return model


@torch.no_grad()
def reconstruct_with_firenet(model, voxel_grid, device):
    """
    Reconstruct image from voxel grid using E2VIDRecurrent
    """
    voxel = voxel_grid.unsqueeze(0).to(device)

    # Normalize voxel grid
    voxel = voxel / (voxel.abs().max() + 1e-8)

    # Pad to multiple of 8 (required for encoder-decoder with 3 encoders)
    _, _, H, W = voxel.shape
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    if pad_h > 0 or pad_w > 0:
        voxel = torch.nn.functional.pad(voxel, (0, pad_w, 0, pad_h), mode='reflect')

    # Forward pass
    output, _ = model(voxel, prev_states=None)

    # Crop back to original size
    output = output[:, :, :H, :W]

    # Convert to uint8 image
    output = output.squeeze().cpu().numpy()
    output = np.clip(output * 255, 0, 255).astype(np.uint8)

    return output


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check if checkpoint exists
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"    ERROR: E2VID checkpoint not found at {CHECKPOINT_PATH}")
        print("\n   Download E2VID pretrained model from:")
        print("     https://github.com/uzh-rpg/rpg_e2vid#pretrained-models")
        print("     Or use the one from your firenet folder if available")
        print(f"\n  Save to: {CHECKPOINT_PATH}")
        return

    # Load FireNet
    model = load_firenet(CHECKPOINT_PATH, device)

    # Import and load dataset
    from spikeclip_snn.data.ncaltech101_dataset import NCaltech101Dataset

    dataset = NCaltech101Dataset(
        root_dir=NCALTECH_ROOT,
        num_bins=5,
    )
    print(f"    Dataset loaded: {len(dataset)} samples")

    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    # Track samples per class for naming
    class_counters = {}

    # Process all samples
    print("\n   Generating HQ reconstructions with FireNet...")
    for idx in tqdm(range(len(dataset))):
        voxel,label = dataset[idx]
        class_name = dataset.classes[label]

        # Initialize counter for this class
        if class_name not in class_counters:
            class_counters[class_name] = 0

        # Reconstruct with FireNet
        hq_image = reconstruct_with_firenet(model, voxel, device)

        # Save to class folder
        class_dir = output_path / class_name
        class_dir.mkdir(exist_ok=True)

        # Save as PNG with index matching dataset
        img_path = class_dir / f"hq_{idx:05d}.png"
        Image.fromarray(hq_image).save(img_path)

        class_counters[class_name] += 1

    # Summary
    print(f"\n  Done! HQ images saved to: {OUTPUT_DIR}")
    print(f"    Total images: {len(dataset)}")
    print(f"    Classes: {len(class_counters)}")


if __name__ == "__main__":
    main()
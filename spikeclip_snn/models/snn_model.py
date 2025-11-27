"""
This file contains SNN-based Image Reconstruction Model
Converts event voxel grids to reconstructed images using Spiking Neural Networks

"""

# IMPORTS
import torch
import torch.nn as nn
import snntorch as snn

from snntorch import surrogate

# SNNRenstruction is the main neural network that
# converts event to images

class SNNReconstruction(nn.Module):
    # Create a model class which we can inherit from nn.Module
    # Input is 5-channel voxel grid
    # Output is 1-channel grayscale image
    """
    Spiking Neural Network for reconstructing images from event voxel grids

    Architecture:
    - Encoder: Compress voxel (5, 180, 240) → features
    - Decoder: Reconstruct features → image (1, 180, 240)
    """

    def __init__(self, num_bins=5, beta=0.9):
        """
        :param num_bins: Number of temporal bins in voxel grid (default: 5)
        :param beta: Decay rate for LIF neurons (default: 0.9)
        """
        super(SNNReconstruction, self).__init__()

        # ENCODER LAYER
        # SNN with LIF neurons

        # Steeper surroagate gradient for better gradient flow
        spike_grad = surrogate.fast_sigmoid(slope=50)
        # ENCODER LAYER

        #   Layer 1, 5 channels to 32
        self.conv1 = nn.Conv2d(num_bins, 64, kernel_size=3, stride=2, padding=1)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        #   Layer 2, 32 channels to 64
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        #   Layer3, 64 channels to 128
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # DECODER LAYER
        #   Layer 4, 128 channels to 64
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.relu1 = nn.ReLU()

        #   Layer 5, 64 channels to 32
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.relu2 = nn.ReLU()

        #   Layer 6, 32 channels to the final image
        self.deconv3 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)
        self.sigmoid = nn.Sigmoid()


    # Forward Method
    def forward(self, x, num_steps=5):
        # Initialize membrane potentials for LIF neurons
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        # Record the membrane potentials
        spk_rec = []

        # Encoder
        # Process voxel through SNN layer
        # Process over multiple timesteps

        for step in range(num_steps):
            # Layer 1, Conv + LIF
            cur1 = self.conv1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            # Layer 2, Conv + LIF
            cur2 = self.conv2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            # Layer 3, Conv + LIF
            cur3 = self.conv3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            # Collect all the final spikes
            spk_rec.append(spk3)

        # Average all timesteps
        # Average over time
        spk_rec = torch.stack(spk_rec, dim=0)
        x = torch.mean(spk_rec, dim=0)

        # Decoder
        # Upsample layer 1 128 to 64 channels
        x = self.deconv1(x)
        x = self.relu1(x)

        #Upsamle layer 2, 64 to 32 channels
        x = self.deconv2(x)
        x = self.relu2(x)

        # Upsample layer 3, 32 to 1 channel
        # Output range [0,  1]
        # This will be the final image
        x = self.deconv3(x)
        x = self.sigmoid(x)

        # Return the reconstructred image
        x = torch.nn.functional.interpolate(x, size=(180, 240), mode='bilinear', align_corners=False)
        return x


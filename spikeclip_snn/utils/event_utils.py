"""
This file contains processing functions for Neuromorphic datasets

Handles conversion of asynchronous event data (x, y, t, polarity) from
neuromorphic cameras (DVS) to representations suitable for SNNs.
"""
import os.path

# Imports
import numpy as np
import torch
from typing import Tuple, Optional
import struct

dtype = np.float32

# We are using N-Caltech101 in this project (https://www.garrickorchard.com/datasets/n-caltech101)
# However, the raw files are unreadible so we need to convert it
# load_events function loads the dataset into .bin file format

def load_events(file_path: str) -> np.ndarray:
    """
    Load events from dataset into .bin format

    N-Caltech101 .bin format is as following;
    Format (40 bits, 5 bytes per event)
    - bit 39-32: X address (8 bits)
    - bit 31-24: Y address (8 bits)
    - bit 23: Polarity (0 = OFF/DARKER, 1 = ON/BRIGHTER) (1 bit)
    - bit 22-0: Timestamp in microseconds (23 bits)

    :param file_path:
    :return:
        events: numpy array shape (N, 4) with columns [x, y, t, polarity]
        N    = number of events
        x    = horizontal pixel position
        y    = vertical pixel position
        t    = timestamp (in ms)
        polarity    = -1 (brightness decreased) +1 (brightness increased)

    """
    # Each event is 5 bytes
    event_size = 5
    events = []

    # Check if file exists, if it doesn't exist raise an error
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Event file not found: {file_path}")

    # Dataset consists of binary files
    # Reading the binary data, rb for binary mode
    # raw data = byte objects
    with open(file_path, "rb") as f:
        raw_data = f.read()

    # Calculating the number of events
    num_events = len(raw_data) // event_size

    # File path could exist, but check if there is a data
    if len(raw_data) == 0:
        raise ValueError(f"Event file is empty : {file_path}")

    # Parse each event
    for i in range(num_events):
        #Get the corresponding 5 bytes for event i
        #Calcualting the offset so i can use in window slice
        offset = i * event_size
        event_bytes = raw_data[offset:offset + event_size]

        # We have 5 bytes for this event
        # Combine the bytes into single chunk 40 bytes
        data = int.from_bytes(event_bytes, byteorder="big")

        # Extracting the fields
        # Bit operations
        # Shift all bits to RIGHT by 32 poistions
        # But even though we get >32 bits there could
        # be potenatial mixes to ensure this
        # A mask of 8 bits introduced

        # data >> 24 = 0b0111100001010000
        #    & 0xFF
        #    0111100001010000  (shifted result)
        #    0000000011111111  (mask)
        #  ──────────────────  &
        #    0000000001010000  (only Y)
        x = (data >> 32) & 0xFF
        y = (data >> 24) & 0xFF
        # bit 23, masking with 0x01
        polarity_bit = (data >> 23) & 0x01
        # No shift to timestamp, only masking
        timestamp = data & 0x7FFFFF
        # Convert polarity to binary
        polarity = 1 if polarity_bit == 1 else -1
        # Add event [x, y, timestamp, [polarity]
        events.append([x,y,timestamp,polarity])
    # Convert the event into NumPy array
    events = np.array(events, dtype = dtype)

    return events

    # Since SNN's need dense input, we need to convert
    # these random list of events to a regular
    # grid/tensor of numbers

def events_to_voxel_grid(events: np.ndarray,
                         num_bins: int,
                         height: int,
                         width: int,
                         normalize: bool = True) -> torch.Tensor:
        """
        Convert events to voxel grid representation

        :param events: numpy array of shape (N, 4) with columns [x, y, t, polarity]
        :param num_bins: number of temporal bins (e.g., 5)
        :param height: height of output grid (180 for N-Caltech101)
        :param width: width of output grid (e.g., 240 for N-Caltech101)
        :param normalize: whether to normalize the voxel grid (default: True)

        :return:
            voxel_grid: torch tensor of shape (num_bins, height, width)
        """

        # Check if events is empty
        if len(events) == 0:
            # If its empty, then return with empty voxel grid
            return torch.zeros(num_bins, height, width, dtype=torch.float32)

        # Extract the x,y, timestamp and polarity from events
        # X coordinates
        x = events[:, 0].astype(np.int32)
        # Y coordinates
        y = events[: ,1].astype(np.int32)
        # timestamp
        t = events[:, 2]
        # polarity
        polarity = events[:, 3]

        # Normalize the timestamps
        # Find the time range
        t_max = t.max()
        t_min = t.min()

        # If time_max == time_min could be division by 0
        if t_max == t_min:
            # All events go to bin 0
            t_norm = np.zeros_like(t)
        else:
            # Normalize to [0, num_bins)
            t_norm = (t - t_min) / (t_max - t_min) * num_bins
            # Clip to valid range
            t_norm = np.clip(t_norm, 0, num_bins - 1)

        # Convert to integer bin indices
        t_idx = t_norm.astype(np.int32)

        # Create an empty voxel
        voxel = np.zeros((num_bins, height, width), dtype=np.float32)

        # Clip the coordinates into valid range
        # Prevents the out-of-bounds errors
        x = np.clip(x, 0, width - 1)
        y = np.clip(y, 0, height - 1)

        # Accumulate each event into vortex grid
        for i in range(len(events)):
            bin_idx = t_idx[i]
            x_pos = x[i]
            y_pos = y[i]
            polarity_value = polarity[i]

            # Add polarity to voxel at [bin, y, x]
            voxel[bin_idx, y_pos, x_pos] += polarity_value

        # Normalize the voxel grid
        if normalize:
            max_val = np.abs(voxel).max()
            if max_val > 0:
                voxel = voxel / max_val

        return voxel







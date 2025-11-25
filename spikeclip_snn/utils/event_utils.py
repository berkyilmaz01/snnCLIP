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

        # Example
        # data >> 24 = 0b0111100001010000
        # & 0xFF
        #
        #    0111100001010000  (shifted result)
        #  & 0000000011111111  (mask)
        #  ──────────────────  AND
        #    0000000001010000  (only Y)
        x = (data >> 32) & 0xFF
        y = (data >> 24) & OxFF
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




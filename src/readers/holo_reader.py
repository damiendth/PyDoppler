import os
import json
from typing import Optional
import numpy as np
import struct

class HoloReader:
    # --- Class-level attribute declarations ---
    filename: str
    all_frames: Optional[np.ndarray]
    version: int
    num_frames: int
    frame_width: int
    frame_height: int
    data_size: int
    bit_depth: int
    endianness: int
    footer: dict

    def __init__(self, filename: str, load_all_file: bool = False) -> None:
        self.filename = filename
        self.all_frames = None

        # --- Parse header ---
        with open(filename, 'rb') as f:
            header_bytes = f.read(29)  # first 29 bytes to read fixed header fields

        # unpack header
        magic_number, version, bit_depth, width, height, num_frames, total_size, endianness = struct.unpack(
            '<4sHHIIIQB', header_bytes
        )

        if magic_number != b'HOLO':
            raise ValueError("Bad holo file.")

        self.version = version
        self.num_frames = num_frames
        self.frame_width = width
        self.frame_height = height
        self.data_size = total_size
        self.bit_depth = bit_depth
        self.endianness = endianness

        # --- Parse footer ---
        footer_skip = 64 + self.frame_width * self.frame_height * self.num_frames * (self.bit_depth // 8)
        file_size = os.path.getsize(filename)
        footer_size = file_size - footer_skip

        with open(filename, 'rb') as f:
            f.seek(footer_skip)
            footer_unparsed = f.read(footer_size)
            self.footer = json.loads(footer_unparsed.decode('utf-8'))

        # Optionally load all frames into RAM
        if load_all_file:
            self.all_frames = self.read_frame_batch(self.num_frames, 1)

    def read_frame_batch(self, batch_size: int, frame_position: int) -> np.ndarray:
        """
        Read a batch of frames from the holo file.

        Parameters:
        - batch_size: number of frames to read
        - frame_position: starting frame position (1-based index)

        Returns:
        - np.ndarray of shape (batch_size, frame_height, frame_width)
        """
        frame_offset = frame_position - 1
        frames = np.zeros((batch_size, self.frame_height, self.frame_width), dtype=np.float32)

        if self.all_frames is not None:
            return self.all_frames[:, :, frame_offset:frame_offset + batch_size]

        frame_size = self.frame_width * self.frame_height * (self.bit_depth // 8)
        endian_char = '<' if self.endianness == 0 else '>'

        retrycnt = 0
        retry = True

        while retry and retrycnt < 3:
            retry = False
            with open(self.filename, 'rb') as f:
                for i in range(batch_size):
                    f.seek(64 + frame_size * (frame_offset + i))
                    try:
                        if self.bit_depth == 8:
                            data = np.frombuffer(f.read(self.frame_width * self.frame_height), dtype=np.uint8)
                        elif self.bit_depth == 16:
                            data = np.frombuffer(f.read(self.frame_width * self.frame_height * 2), dtype=endian_char + 'u2')
                        else:
                            raise ValueError(f"Unsupported bit depth: {self.bit_depth}")

                        frames[i, :, :] = data.reshape((self.frame_height, self.frame_width))

                    except Exception as e:
                        retry = True
                        retrycnt += 1
                        print(f"Holo file frame in position {i+1} was not found: {e}")
                        frames[i, :, :] = np.nan

        return frames

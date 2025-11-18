import numpy as np
from settings.settings import Settings
import matplotlib.pyplot as plt


def batch_average(frames: np.ndarray, start: int = 0, end: int = -1) -> np.ndarray:
    if start < 0:
        start = 0
    if end == -1 or end > frames.shape[0]:
        end = frames.shape[0]

    avg = np.mean(np.abs(frames[start:end]), axis=0)
    return (avg - np.min(avg)) / (np.max(avg) - np.min(avg)) * 255


def sliding_average(frames: np.ndarray, settings: Settings) -> np.ndarray:
    output_frames = []

    for i in range(frames.shape[0]):

        output_frames.append(
            batch_average(frames, start=i, end=i + settings.sliding_average_window_size)
        )

    return np.stack(output_frames)

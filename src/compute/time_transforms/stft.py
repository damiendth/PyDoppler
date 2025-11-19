import numpy as np

from settings.settings import Settings


def stft(frames: np.ndarray, settings: Settings) -> np.ndarray:
    # print("Computing STFT Transform H shape:", frames.shape)
    return np.fft.fft(frames, axis=0)

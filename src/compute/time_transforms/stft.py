import numpy as np
def stft(frames: np.ndarray) -> np.ndarray:

    return np.fft.fft(frames, axis=0)

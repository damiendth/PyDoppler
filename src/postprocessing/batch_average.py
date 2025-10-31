import numpy as np

def batch_average(frames: np.ndarray, start: int = 0, end: int = -1) -> np.ndarray:
    if end == -1:
        end = frames.shape[0]
    return np.mean(np.abs(frames[start:end]), axis=0)
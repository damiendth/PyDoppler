import cv2
import numpy as np

def normalize_frames(frames: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(frames):
        frames = np.abs(frames)

    normalized = []
    for el in frames:
        norm_frame = cv2.normalize(el, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) # type: ignore
        normalized.append(norm_frame)

    return np.stack(normalized)

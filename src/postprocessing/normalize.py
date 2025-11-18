import cv2
import numpy as np

from settings.settings import Settings


def normalize_frames(frames: np.ndarray, settings: Settings) -> np.ndarray:
    if np.iscomplexobj(frames):
        frames = np.abs(frames)

    normalized = []
    for el in frames:
        norm_frame = cv2.normalize(el, None, 0, 255, cv2.NORM_MINMAX) # type: ignore
        normalized.append(norm_frame)

    return np.stack(normalized)

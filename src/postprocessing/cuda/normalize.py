import cupy as cp
from typing import Optional
from settings.settings import Settings

def normalize_frames_cuda(
    frames: cp.ndarray, 
    settings: Settings, 
    stream: Optional[cp.cuda.Stream] = None
) -> cp.ndarray:
    """
    Normalize each frame to [0, 255] entirely on GPU, fully non-blocking for CPU.
    Behaves like the original cv2.normalize loop but vectorized on GPU.

    Parameters:
    - frames: cp.ndarray, shape (nbframes, H, W), can be real or complex
    - settings: Settings object (kept for interface)
    - stream: optional CuPy stream for asynchronous execution

    Returns:
    - normalized_frames: cp.ndarray, same shape as frames, dtype=float32
    """
    stream = stream or cp.cuda.Stream.null

    with stream: # type: ignore
        frames_abs = cp.abs(frames) if cp.iscomplexobj(frames) else frames

        mins = frames_abs.min(axis=(1, 2), keepdims=True)
        maxs = frames_abs.max(axis=(1, 2), keepdims=True)
        ranges = cp.where(maxs - mins == 0, 1, maxs - mins)

        normalized_frames = 255 * (frames_abs - mins) / ranges
        # normalized_frames = normalized_frames.astype(cp.float32)

    return normalized_frames

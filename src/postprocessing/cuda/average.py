import cupy as cp
from typing import Optional
from settings.settings import Settings


def batch_average_gpu(
    frames: cp.ndarray,
    start: int = 0,
    end: int = -1,
    stream: Optional[cp.cuda.Stream] = None,
) -> cp.ndarray:
    """
    Compute batch average of frames[start:end], normalize to [0,255], fully on GPU.
    """
    stream = stream or cp.cuda.Stream.null
    with stream:  # type: ignore
        n_frames = frames.shape[0]
        if start < 0:
            start = 0
        if end == -1 or end > n_frames:
            end = n_frames

        batch = frames[start:end]
        batch_abs = cp.abs(batch)

        avg = cp.mean(batch_abs, axis=0)

        min_val = cp.min(avg)
        max_val = cp.max(avg)
        range_val = max_val - min_val
        range_val = cp.where(range_val == 0, 1, range_val)

        normalized_avg = 255 * (avg - min_val) / range_val
    return normalized_avg


def sliding_average_gpu(
    frames: cp.ndarray, settings: Settings, stream: Optional[cp.cuda.Stream] = None
) -> cp.ndarray:
    """
    Apply a sliding average over frames with window size from settings.
    Fully GPU-resident and non-blocking.
    """
    stream = stream or cp.cuda.Stream.null
    window_size = settings.sliding_average_window_size
    n_frames = frames.shape[0]

    output_frames = []

    with stream:  # type: ignore
        for i in range(n_frames):
            end = min(i + window_size, n_frames)
            avg_frame = batch_average_gpu(frames, start=i, end=end, stream=stream)
            output_frames.append(avg_frame)

        output_frames = cp.stack(output_frames)

    return output_frames

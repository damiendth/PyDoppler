import cupy as cp
from settings.settings import Settings

def stft_cuda(frames: cp.ndarray, settings: Settings, stream: cp.cuda.Stream = None) -> cp.ndarray:
    """
    Compute the STFT (FFT along axis=0) entirely on GPU.

    Parameters:
    - frames: cp.ndarray, shape (nbframes, height, width) or similar
    - settings: Settings object (not used here, kept for interface)
    - stream: cp.cuda.Stream, optional, for asynchronous execution

    Returns:
    - frames_fft: cp.ndarray, same shape as frames, complex64/complex128
    """
    stream = stream or cp.cuda.Stream.null

    with stream:
        frames_fft = cp.fft.fft(frames, axis=0)

    return frames_fft

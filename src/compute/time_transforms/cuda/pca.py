import cupy as cp
from settings.settings import Settings

def pca_cuda(frames: cp.ndarray, settings: Settings, stream: cp.cuda.Stream = None) -> cp.ndarray:
    """
    Perform PCA on a batch of frames entirely on GPU using CuPy.

    Parameters:
    - frames: cp.ndarray, shape (nbframes, height, width), complex64/complex128
    - settings: Settings object (not used here, but kept for interface)
    - stream: cp.cuda.Stream, optional GPU stream

    Returns:
    - transformed_frames: cp.ndarray, shape same as frames
    """

    stream = stream or cp.cuda.Stream.null

    with stream:
        c, h, w = frames.shape
        H = frames.reshape(c, h * w).T 

        cov_matrix = cp.cov(H, rowvar=False) 

        eigenvalues, eigenvectors = cp.linalg.eigh(cov_matrix)

        sorted_indices = cp.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        H_transformed = H @ sorted_eigenvectors 

        transformed_frames = H_transformed.T.reshape(c, h, w)

    return transformed_frames

import numpy as np

from settings.settings import Settings


def pca(frames: np.ndarray, settings: Settings) -> np.ndarray:
    H = frames
    c, h, w = H.shape
    # print("Computing PCA Transform H shape:", H.shape)
    H = H.reshape(c, h * w).T
    cov_matrix = np.cov(H, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    H = H @ sorted_eigenvectors
    return H.T.reshape(c, h, w)

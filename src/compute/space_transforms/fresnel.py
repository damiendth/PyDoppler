import numpy as np
from typing import Tuple

def fresnel_kernel(
    Nx: int,
    Ny: int,
    z: float,
    wavelength: float,
    x_step: float,
    y_step: float,
    use_double_precision: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Fresnel propagation kernel and phase factor.
    """
    print(f"Computing Fresnel kernel with Nx={Nx}, Ny={Ny}, z_propagation_distance={z}, wavelength={wavelength}, x_step={x_step}, y_step={y_step}, use_double_precision={use_double_precision}")

    Nx  = float(Nx) # type: ignore
    Ny  = float(Ny) # type: ignore
    x = (np.arange(Nx) - np.round(Nx / 2)) * x_step
    y = (np.arange(Ny) - np.round(Ny / 2)) * y_step
    X, Y = np.meshgrid(x, y)

    kernel = np.exp(1j * np.pi / (wavelength * z) * (X**2 + Y**2))

    phase_factor = (
        1j / wavelength
        * np.exp(-2j * np.pi * z / wavelength)
        * np.exp(
            -1j
            * np.pi
            * (
                (X * wavelength * z / (x_step**2 * Nx)) ** 2
                + (Y * wavelength * z / (y_step**2 * Ny)) ** 2
            )
            / (wavelength * z)
        )
    )

    if not use_double_precision:
        kernel = kernel.astype(np.complex64)
        phase_factor = phase_factor.astype(np.complex64)

    return kernel, phase_factor


def fresnel_transform(
    frames: np.ndarray,
    z: float,
    wavelength: float,
    x_step: float,
    y_step: float,
    use_double_precision: bool = False
) -> np.ndarray:
    """
    Apply Fresnel propagation to a batch of complex frames using a precomputed kernel and phase factor.

    Parameters:
    - frames: np.ndarray, shape (nbframes, Ny, Nx)
        Complex-valued input frames.
    - z: float
        Propagation distance in meters.
    - wavelength: float
        Wavelength of light in meters.
    - x_step: float
        Pixel pitch along the x-axis (meters per pixel).
    - y_step: float
        Pixel pitch along the y-axis (meters per pixel).
    - use_double_precision: bool, optional
        If True, uses complex128 precision; otherwise complex64.

    Returns:
    - frames_out: np.ndarray, same shape as `frames`
        Fresnel-propagated complex frames.
    """

    nbframes, Ny, Nx = frames.shape

    kernel, phase_factor = fresnel_kernel(
        Nx, Ny,
        z,
        wavelength,
        x_step,
        y_step,
        use_double_precision=use_double_precision
    )

    frames_out = np.fft.fft2(frames * kernel, axes=(1, 2))
    frames_out *= phase_factor

    return frames_out

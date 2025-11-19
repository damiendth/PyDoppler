import cupy as cp
from typing import Tuple
from settings.settings import Settings

def fresnel_kernel_cuda(
    Nx: int,
    Ny: int,
    z: float,
    wavelength: float,
    x_step: float,
    y_step: float,
    use_double_precision: bool = True,
    stream: cp.cuda.Stream = None
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Compute the Fresnel propagation kernel and phase factor on GPU.
    """

    Nx = float(Nx) # type: ignore
    Ny = float(Ny) # type: ignore

    # Use the provided stream
    stream = stream or cp.cuda.Stream.null

    with stream:
        x = (cp.arange(Nx) - cp.round(Nx / 2)) * x_step
        y = (cp.arange(Ny) - cp.round(Ny / 2)) * y_step
        X, Y = cp.meshgrid(x, y)

        kernel = cp.exp(1j * cp.pi / (wavelength * z) * (X**2 + Y**2))

        phase_factor = (
            1j / wavelength
            * cp.exp(-2j * cp.pi * z / wavelength)
            * cp.exp(
                -1j
                * cp.pi
                * (
                    (X * wavelength * z / (x_step**2 * Nx)) ** 2
                    + (Y * wavelength * z / (y_step**2 * Ny)) ** 2
                )
                / (wavelength * z)
            )
        )

        if not use_double_precision:
            kernel = kernel.astype(cp.complex64)
            phase_factor = phase_factor.astype(cp.complex64)

    return kernel, phase_factor


def fresnel_transform_cuda(
    frames: cp.ndarray,
    settings: Settings,
    stream: cp.cuda.Stream = None
) -> cp.ndarray:
    """
    Apply Fresnel propagation to a batch of complex frames using GPU arrays.
    Fully GPU-resident, asynchronous if a stream is provided.

    Parameters:
    - frames: cp.ndarray, shape (nbframes, Ny, Nx)
    - settings: Settings object
    - stream: cp.cuda.Stream, optional

    Returns:
    - frames_out: cp.ndarray, same shape as `frames`
    """

    nbframes, Ny, Nx = frames.shape

    z = settings.space_transform.z
    wavelength = settings.space_transform.wavelength
    x_step = settings.space_transform.x_step
    y_step = settings.space_transform.y_step
    use_double_precision = settings.space_transform.use_double_precision
    shift_after = settings.space_transform.shift_after

    stream = stream or cp.cuda.Stream.null

    # Compute kernel and phase factor on GPU in the given stream
    kernel, phase_factor = fresnel_kernel_cuda(
        Nx, Ny, z, wavelength, x_step, y_step,
        use_double_precision=use_double_precision,
        stream=stream
    )

    with stream:

        frames_out = cp.fft.fft2(frames * kernel, axes=(1, 2))
        frames_out *= phase_factor

        if shift_after:
            frames_out = cp.fft.fftshift(frames_out, axes=(-2, -1))

    return frames_out

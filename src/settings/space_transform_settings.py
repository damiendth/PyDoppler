from enums import TransformType


class SpaceTransformSettings:

    z: float  # Propagation distance in meters
    wavelength: float  # Wavelength in meters
    x_step: float  # Pixel pitch in x direction in meters
    y_step: float  # Pixel pitch in y direction in meters
    shift_after: bool  # Whether to apply fftshift after Fresnel transform
    transform_type: TransformType  # Type of space transform to apply

    def __init__(
        self,
        z: float = 0.0,
        wavelength: float = 0.0,
        x_step: float = 0.0,
        y_step: float = 0.0,

        shift_after: bool = False,
        transform_type: TransformType = TransformType.NONE,
    ) -> None:
        self.z = z
        self.wavelength = wavelength
        self.x_step = x_step
        self.y_step = y_step
        self.shift_after = shift_after
        self.transform_type = transform_type

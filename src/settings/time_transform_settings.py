from enums import TransformType


class TimeTransformSettings:
    z_stft: int  # STFT batch start index along time axis
    width_stft: int  # STFT window size
    transform_type: TransformType  # Type of time transform to apply

    def __init__(
        self,
        z_stft: int = 0,
        width_stft: int = 0,
        transform_type: TransformType = TransformType.STFT,
    ) -> None:
        self.z_stft = z_stft
        self.width_stft = width_stft
        self.transform_type = transform_type

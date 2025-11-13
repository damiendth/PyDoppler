from enums import TransformType


class TimeTransformSettings:
    z_stft: int  # STFT batch start index along time axis
    window_size: int  # STFT window size
    transform_type: TransformType  # Type of time transform to apply

from settings.space_transform_settings import SpaceTransformSettings
from settings.time_transform_settings import TimeTransformSettings
from enums import TransformType


class Settings:

    space_transform: SpaceTransformSettings = SpaceTransformSettings()
    time_transform: TimeTransformSettings = TimeTransformSettings()

    batch_size: int
    batch_stride: int
    num_workers: int
    sliding_average_window_size: int

    use_double_precision: bool
    use_cuda: bool
    output_video_name: str = "output_video"

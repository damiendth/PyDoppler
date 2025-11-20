from settings.space_transform_settings import SpaceTransformSettings
from settings.time_transform_settings import TimeTransformSettings
from enums import TransformType


class Settings:

    space_transform: SpaceTransformSettings
    time_transform: TimeTransformSettings

    batch_size: int
    batch_stride: int
    num_workers: int
    num_gpu_workers: int
    sliding_average_window_size: int

    use_double_precision: bool
    use_cuda: bool
    output_video_name: str

    def __init__(
        self,
        space_transform: SpaceTransformSettings = SpaceTransformSettings(),
        time_transform: TimeTransformSettings = TimeTransformSettings(),
        batch_size: int = 0,
        batch_stride: int = 0,
        num_workers: int = 0,
        num_gpu_workers: int = 0,
        sliding_average_window_size: int = 0,
        use_double_precision: bool = False,
        use_cuda: bool = False,
        output_video_name: str = "output_video",
    ) -> None:
        self.space_transform = space_transform
        self.time_transform = time_transform
        self.batch_size = batch_size
        self.batch_stride = batch_stride
        self.num_workers = num_workers
        self.num_gpu_workers = num_gpu_workers
        self.sliding_average_window_size = sliding_average_window_size
        self.use_double_precision = use_double_precision
        self.use_cuda = use_cuda
        self.output_video_name = output_video_name

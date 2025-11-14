from fileinput import filename
import numpy as np
import matplotlib.pyplot as plt
from enums import TransformType
from exec import Executor
from readers.holo_reader import HoloReader
from settings.settings import Settings
from settings.space_transform_settings import SpaceTransformSettings
from settings.time_transform_settings import TimeTransformSettings
from writers.video_writer import write_video
from compute.space_transforms.fresnel import fresnel_transform
from compute.time_transforms.pca import pca


def main() -> None:

    filename = "E:\\250512\\250512_GUJ_L.holo"

    settings = Settings()
    reader = HoloReader(filename, load_all_file=False)
    executor = Executor(settings=settings, reader=reader)

    settings.batch_size = 32
    settings.batch_stride = 32

    settings.sliding_average_window_size = 32
    settings.use_double_precision = False

    settings.space_transform = SpaceTransformSettings(
        z=reader.footer["compute_settings"]["image_rendering"]["propagation_distance"],
        wavelength=reader.footer["compute_settings"]["image_rendering"]["lambda"],
        x_step=reader.footer["info"]["pixel_pitch"]["x"] * 1e-6,
        y_step=reader.footer["info"]["pixel_pitch"]["y"] * 1e-6,
        use_double_precision=False,
        shift_after=True,
        transform_type=TransformType.FRESNEL
    )

    settings.time_transform.transform_type = TransformType.PCA

    filename = "E:\\250512\\250512_GUJ_L.holo"
    reader = HoloReader(filename, load_all_file=False)

    executor.build_pipe()
    executor.execute()



if __name__ == "__main__":
    main()

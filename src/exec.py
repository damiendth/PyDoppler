import numpy as np
import matplotlib.pyplot as plt
from compute.time_transforms.stft import stft
from enums import TransformType
from postprocessing.sliding_average import batch_average, sliding_average
from postprocessing.normalize import normalize_frames
from readers.holo_reader import HoloReader
from settings.settings import Settings
from writers.video_writer import write_video
from compute.space_transforms.fresnel import fresnel_transform
from compute.time_transforms.pca import pca


class Executor:

    settings: Settings
    reader: HoloReader

    _pipeline: list

    def __init__(self, settings: Settings, reader: HoloReader) -> None:
        self.settings = settings
        self.reader = reader
        self._pipeline = []

    def build_pipe(self) -> None:

        if self.settings.space_transform.transform_type == TransformType.FRESNEL:
            self._pipeline.append(fresnel_transform)

        if self.settings.time_transform.transform_type == TransformType.STFT:
            self._pipeline.append(stft)
        elif self.settings.time_transform.transform_type == TransformType.PCA:
            self._pipeline.append(pca)

        self._pipeline.append(normalize_frames)

    def clear_pipe(self) -> None:
        self._pipeline = []

    def run_pipeline(self, batch: np.ndarray) -> np.ndarray:

        for step in self._pipeline:
            batch = step(frames=batch, settings=self.settings)

        return batch

    def execute(self) -> None:

        total = []
        
        for i in range(0, self.reader.num_frames, self.settings.batch_stride):
            if i // self.settings.batch_stride == 32:
                break
            print(f"Processing batch number {i // self.settings.batch_stride + 1}...")
            
            batch = self.reader.read_frame_batch(
                batch_size=self.settings.batch_size, frame_position=i
            )

            processed_batch = self.run_pipeline(batch)
            avg_frame = batch_average(processed_batch)
            total.append(avg_frame)
        
        print("Writing output video...")
        sliding_average_frames = sliding_average(np.array(total), self.settings)
        write_video(
            sliding_average_frames,
            f"{self.settings.output_video_name}_batch_{1}",
            fps=1,
            format="avi",
        )

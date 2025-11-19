import os
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
import cupy as cp
from concurrent.futures import ThreadPoolExecutor


class Executor:

    settings: Settings
    reader: HoloReader

    _pipeline: list
    _cuda_streams: list

    _use_cuda: bool

    def __init__(
        self, settings: Settings, reader: HoloReader, use_cuda: bool = False
    ) -> None:
        self.settings = settings
        self.reader = reader
        self._pipeline = []
        self._cuda_streams = []
        self._use_cuda = use_cuda

        if use_cuda:
            for _ in range(
                reader.num_frames // settings.batch_size
                if reader.num_frames % settings.batch_size == 0
                else reader.num_frames // settings.batch_size + 1
            ):
                self._cuda_streams.append(cp.cuda.Stream())

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
        print("Starting processing...")
        
        def worker(batch_index: int) -> np.ndarray:
            print(f"Processing batch number {batch_index + 1}...")

            frame_position = batch_index * self.settings.batch_stride
            batch = self.reader.read_frame_batch(
                batch_size=self.settings.batch_size, frame_position=frame_position
            )

            processed_batch = self.run_pipeline(batch)
            avg_frame = batch_average(processed_batch)
            return avg_frame

        thread_num = (
            self.settings.num_workers
            if self.settings.num_workers > 0
            else os.cpu_count()
        )

        with ThreadPoolExecutor(max_workers=thread_num) as executor:
            futures = []
            num_batches = (
                self.reader.num_frames // self.settings.batch_size
                if self.reader.num_frames % self.settings.batch_size == 0
                else self.reader.num_frames // self.settings.batch_size + 1
            )

            for batch_index in range(num_batches):
                futures.append(executor.submit(worker, batch_index))

            total = [future.result() for future in futures]

        print("Writing output video...")

        sliding_average_frames = sliding_average(np.array(total), self.settings)

        plt.imshow(sliding_average_frames[0], cmap='gray')
        plt.title("Final Average Frame")
        plt.show()

        write_video(
            sliding_average_frames,
            f"{self.settings.output_video_name}_batch_{1}",
            fps=1,
            format="avi",
        )

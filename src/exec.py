import os
import numpy as np
import matplotlib.pyplot as plt
from compute.space_transforms.cuda.fresnel import fresnel_transform_cuda
from compute.time_transforms.cuda.pca import pca_cuda
from compute.time_transforms.cuda.stft import stft_cuda
from compute.time_transforms.stft import stft
from enums import TransformType
from postprocessing.cuda.average import batch_average_gpu, sliding_average_gpu
from postprocessing.cuda.normalize import normalize_frames_cuda
from postprocessing.average import batch_average, sliding_average
from postprocessing.normalize import normalize_frames
from readers.holo_reader import HoloReader
from settings.settings import Settings
from utility import print_frame
from writers.video_writer import write_video
from compute.space_transforms.fresnel import fresnel_transform
from compute.time_transforms.pca import pca
import cupy as cp
from concurrent.futures import ThreadPoolExecutor
import time


class Executor:

    settings: Settings
    reader: HoloReader

    _pipeline: list
    _cuda_pipeline: list
    _cuda_streams: list

    def __init__(self, settings: Settings, reader: HoloReader) -> None:
        self.settings = settings
        self.reader = reader
        self._pipeline = []
        self._cuda_pipeline = []
        self._cuda_streams = []

    def build_pipe(self) -> None:

        if not self.settings.use_cuda:
            if self.settings.space_transform.transform_type == TransformType.FRESNEL:
                self._pipeline.append(fresnel_transform)
            if self.settings.time_transform.transform_type == TransformType.STFT:
                self._pipeline.append(stft)
            elif self.settings.time_transform.transform_type == TransformType.PCA:
                self._pipeline.append(pca)
            self._pipeline.append(normalize_frames)

        else:
            if self.settings.space_transform.transform_type == TransformType.FRESNEL:
                self._cuda_pipeline.append(fresnel_transform_cuda)
            if self.settings.time_transform.transform_type == TransformType.STFT:
                self._cuda_pipeline.append(stft_cuda)
            elif self.settings.time_transform.transform_type == TransformType.PCA:
                self._cuda_pipeline.append(pca_cuda)
            self._cuda_pipeline.append(normalize_frames_cuda)

    def clear_pipe(self) -> None:
        self._pipeline = []

    def run_pipeline(self, batch: np.ndarray) -> np.ndarray:

        for step in self._pipeline:
            batch = step(frames=batch, settings=self.settings)

        return batch

    def execute_gpu(self) -> None:

        def worker_gpu(batch_index: int, stream_index: int) -> np.ndarray:
            print(f"Processing batch number {batch_index + 1} on GPU...")

            frame_position = batch_index * self.settings.batch_stride
            batch = self.reader.read_frame_batch(
                batch_size=self.settings.batch_size, frame_position=frame_position
            )

            current_stream = self._cuda_streams[stream_index]

            with current_stream:
                batch_gpu = cp.asarray(batch)

                if self.settings.use_double_precision:
                    batch_gpu = batch_gpu.astype(cp.float64)

                for step in self._cuda_pipeline:
                    batch_gpu = step(frames=batch_gpu, settings=self.settings)

            avg_frame = batch_average_gpu(batch_gpu, stream=current_stream)
            del batch_gpu

            return avg_frame

        print("Starting processing on GPU...")
        first_time = time.time()

        stream_num = (
            self.settings.num_gpu_workers if self.settings.num_gpu_workers > 0 else 4
        )
        self._cuda_streams = [cp.cuda.Stream() for _ in range(stream_num)]

        num_batches = (
            self.reader.num_frames // self.settings.batch_size
            if self.reader.num_frames % self.settings.batch_size == 0
            else self.reader.num_frames // self.settings.batch_size + 1
        )
        total = []

        for batch_index in range(num_batches):
            stream_index = batch_index % stream_num
            self._cuda_streams[stream_index].synchronize()
            avg_frame = worker_gpu(batch_index, stream_index)
            total.append(avg_frame)

        for stream in self._cuda_streams:
            stream.synchronize()

        middle_time = time.time()
        print(f"GPU processing time: {middle_time - first_time} seconds")
        print("Applying sliding average...")

        total_gpu = cp.stack(total)
        sliding_average_frames = sliding_average_gpu(total_gpu, self.settings)
        sliding_average_frames = cp.asnumpy(sliding_average_frames)
        print(f"sliding average time: {time.time() - middle_time} seconds")
        # print_frame(sliding_average_frames[0])

        print("Writing output video...")
        write_video(
            sliding_average_frames,
            f"{self.settings.output_video_name}_batch_{1}",
            fps=30,
            format="avi",
        )
        end_time = time.time()
        print(f"Total execution time: {end_time - first_time} seconds")

    def execute(self) -> None:
        def worker_cpu(batch_index: int) -> np.ndarray:
            print(f"Processing batch number {batch_index + 1} on CPU...")

            frame_position = batch_index * self.settings.batch_stride
            batch = self.reader.read_frame_batch(
                batch_size=self.settings.batch_size, frame_position=frame_position
            )

            processed_batch = self.run_pipeline(batch)
            avg_frame = batch_average(processed_batch)
            return avg_frame

        print("Starting processing on CPU...")

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
                worker = worker_cpu
                futures.append(executor.submit(worker, batch_index))

            total = [future.result() for future in futures]

        print("Applying sliding average...")
        sliding_average_frames = sliding_average(np.array(total), self.settings)

        print_frame(sliding_average_frames[0])

        print("Writing output video...")
        write_video(
            sliding_average_frames,
            f"{self.settings.output_video_name}_batch_{1}",
            fps=30,
            format="avi",
        )

import numpy as np
import matplotlib.pyplot as plt
from readers.holo_reader import HoloReader
from writers.video_writer import write_video
from compute.space_transforms.fresnel import fresnel_transform
from compute.time_transforms.pca import pca
def main() -> None:
    filename = "E:\\250512\\250512_GUJ_L.holo"
    reader = HoloReader(filename, load_all_file=False)

    print("Frame dimensions:", reader.frame_width, "x", reader.frame_height)

    batch = reader.read_frame_batch(batch_size=150, frame_position=1)

    batch = fresnel_transform(
        frames=batch,
        z=reader.footer['compute_settings']['image_rendering']['propagation_distance'],
        wavelength=reader.footer['compute_settings']['image_rendering']['lambda'],
        x_step=reader.footer['info']['pixel_pitch']['x'] * 1e-6,
        y_step=reader.footer['info']['pixel_pitch']['y'] * 1e-6,
        use_double_precision=False
    )
    
    batch = np.fft.fftshift(batch, axes=(-2, -1))

    batch = pca(batch)

    write_video(batch  , "output_video", fps=30, format='avi')

    first_frame = batch[0]
    plt.imshow(np.abs(first_frame), cmap='gray')
    plt.title("First Frame")
    plt.show()



if __name__ == "__main__":
    main()
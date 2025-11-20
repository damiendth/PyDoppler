import matplotlib.pyplot as plt
import numpy as np

def print_frame(frame: np.ndarray) -> None:

    plt.imshow(frame, cmap="gray")
    plt.title("Frame Display")
    plt.show()
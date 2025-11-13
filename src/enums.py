from enum import Enum


class TransformType(Enum):
    NONE = -1
    FRESNEL = 0
    ANGULAR_SPECTRUM = 1
    STFT = 2
    PCA = 3

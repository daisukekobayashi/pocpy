import cv2
import numpy as np
import pytest
from scipy import ndimage

SIZE = 512
CROP = 128


def crop(img):
    c = SIZE // 2
    return img[c - CROP : c + CROP, c - CROP : c + CROP]


def shift(img, dy, dx):
    """Shift `img` by a (sub-pixel) amount with periodic boundaries."""
    spectrum = ndimage.fourier_shift(np.fft.fft2(img), (dy, dx))
    return np.real(np.fft.ifft2(spectrum))


def rotate_scale(img, angle, scale):
    """Rotate counter-clockwise by `angle` degrees and scale."""
    m = cv2.getRotationMatrix2D((SIZE / 2, SIZE / 2), angle, scale)
    return cv2.warpAffine(img, m, (SIZE, SIZE), flags=cv2.INTER_CUBIC)


@pytest.fixture(scope="session")
def texture():
    rng = np.random.default_rng(0)
    return ndimage.gaussian_filter(rng.random((SIZE, SIZE)), 3)

import cv2
import cv2.cv as cv
import numpy as np


def logpolar(src, center, magnitude_scale=40):

    mat1 = cv.fromarray(np.float64(src))
    mat2 = cv.CreateMat(src.shape[0], src.shape[1], mat1.type)

    cv.LogPolar(
        mat1,
        mat2,
        center,
        magnitude_scale,
        cv.CV_INTER_CUBIC + cv.CV_WARP_FILL_OUTLIERS,
    )

    return np.asarray(mat2)

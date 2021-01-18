import cv2


def logpolar(src, center, magnitude_scale=40):
    return cv2.logPolar(
        src, center, magnitude_scale, cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS
    )

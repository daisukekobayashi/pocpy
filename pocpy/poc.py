from collections.abc import Callable

import cv2
import numpy as np
from numpy import pi, sin
from scipy import fft
from scipy.optimize import leastsq


def logpolar(src, center, magnitude_scale=40):
    # Equivalent to the removed cv2.logPolar: rho = M * log(r)
    return cv2.warpPolar(
        src,
        (src.shape[1], src.shape[0]),
        center,
        np.exp(src.shape[1] / magnitude_scale),
        cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LOG,
    )


def zero_padding(src, dstshape, pos=(0, 0)):
    y, x = pos
    dst = np.zeros(dstshape)
    dst[y : src.shape[0] + y, x : src.shape[1] + x] = src
    return dst


def pocfunc_model(alpha, delta1, delta2, r, u):
    N1, N2 = r.shape
    V1, V2 = (2 * x + 1 for x in u)
    return lambda n1, n2: (
        alpha
        / (N1 * N2)
        * sin((n1 + delta1) * V1 / N1 * pi)
        * sin((n2 + delta2) * V2 / N2 * pi)
        / (sin((n1 + delta1) * pi / N1) * sin((n2 + delta2) * pi / N2))
    )


def pocfunc(
    f: np.ndarray,
    g: np.ndarray,
    withlpf: bool = False,
    windowfunc: Callable[[int], np.ndarray] = np.hanning,
) -> np.ndarray:
    """Return the phase-only correlation surface of `f` and `g`."""
    m = np.floor(np.array(f.shape) / 2.0)
    u = m / 2.0

    # hanning window
    hy = windowfunc(f.shape[0])
    hx = windowfunc(f.shape[1])
    hw = hy.reshape(hy.shape[0], 1) * hx
    f = f * hw
    g = g * hw

    # compute 2d fft
    F = fft.fft2(f)
    G = fft.fft2(g)
    G_ = np.conj(G)
    R = F * G_ / np.abs(F * G_)

    if withlpf:
        R = fft.fftshift(R)
        lpf = np.ones((m + 1).astype(int))
        lpf = zero_padding(lpf, f.shape, u.astype(int))
        R = R * lpf
        R = fft.fftshift(R)

    return fft.fftshift(np.real(fft.ifft2(R)))


def poc(
    f: np.ndarray,
    g: np.ndarray,
    withlpf: bool = False,
    fitting_shape: tuple[int, int] = (9, 9),
) -> tuple[float, float, float]:
    """Estimate the translation of `g` relative to `f`.

    Returns `(peak, dy, dx)` with sub-pixel accuracy.
    """
    m = np.floor(np.array(f.shape) / 2.0)
    u = m / 2.0

    r = pocfunc(f, g, withlpf=withlpf)

    # least-square fitting
    peak = np.unravel_index(np.argmax(r), r.shape)

    mf = [x // 2 for x in fitting_shape]
    # r is periodic, so wrap around the border instead of truncating
    rows = np.arange(peak[0] - mf[0], peak[0] + mf[0] + 1) % r.shape[0]
    cols = np.arange(peak[1] - mf[1], peak[1] + mf[1] + 1) % r.shape[1]
    fitting_area = r[np.ix_(rows, cols)]

    p0 = [0.5, -(peak[0] - m[0]) - 0.02, -(peak[1] - m[1]) - 0.02]
    y, x = np.mgrid[-mf[0] : mf[0] + 1, -mf[1] : mf[1] + 1]
    y = y + peak[0] - m[0]
    x = x + peak[1] - m[1]

    def errorfunction(p):
        model = pocfunc_model(p[0], p[1], p[2], r, u)
        return np.ravel(model(y, x) - fitting_area)

    plsq = leastsq(errorfunction, p0)
    return (plsq[0][0], plsq[0][1], plsq[0][2])


def ripoc(
    f: np.ndarray,
    g: np.ndarray,
    withlpf: bool = False,
    fitting_shape: tuple[int, int] = (9, 9),
    M: float | None = None,
) -> tuple[float, float, float, float, float]:
    """Estimate translation, rotation and scale of `g` relative to `f`.

    Returns `(peak, dy, dx, angle, scale)`. `angle` is in degrees.
    `M` is the log-polar magnitude scale; by default it is chosen so
    that the spectrum fills the log-polar image.
    """
    if f.shape != g.shape:
        raise ValueError(
            f"f and g must have the same shape, got {f.shape} and {g.shape}"
        )

    # zero-pad to a square so both axes share the same frequency step
    n = max(f.shape)
    if M is None:
        M = n / np.log(n / 2)

    hy = np.hanning(f.shape[0])
    hx = np.hanning(f.shape[1])
    hw = hy.reshape(hy.shape[0], 1) * hx

    ff = f * hw
    gg = g * hw

    F = fft.fft2(ff, s=(n, n))
    G = fft.fft2(gg, s=(n, n))

    F = fft.fftshift(np.log(np.abs(F)))
    G = fft.fftshift(np.log(np.abs(G)))

    FLP = logpolar(F, (n / 2, n / 2), M)
    GLP = logpolar(G, (n / 2, n / 2), M)

    R = poc(FLP, GLP, withlpf=withlpf, fitting_shape=fitting_shape)

    angle = -R[1] / n * 360
    # logpolar maps radius r to M * log(r)
    scale = np.exp(-R[2] / M)

    # OpenCV takes the center as (x, y)
    center = (g.shape[1] / 2, g.shape[0] / 2)
    rot = cv2.getRotationMatrix2D(center, -angle, 1.0 / scale)

    g_dash = cv2.warpAffine(
        g, rot, (g.shape[1], g.shape[0]), flags=cv2.INTER_LANCZOS4
    )

    t = poc(f, g_dash, withlpf=withlpf, fitting_shape=fitting_shape)

    return (t[0], t[1], t[2], angle, scale)

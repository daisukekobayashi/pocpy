import sys

import numpy as np
from numpy import pi, sin, cos
from scipy.optimize import leastsq
import scipy, scipy.fftpack
import six

import cv2

if cv2.__version__[0] == "2":
    import cv2.cv as cv
    from pocpy.logpolar_opencv2 import *
else:
    from pocpy.logpolar_opencv3 import *


def zero_padding(src, dstshape, pos=(0, 0)):
    y, x = pos
    dst = np.zeros(dstshape)
    dst[y : src.shape[0] + y, x : src.shape[1] + x] = src
    return dst


def pocfunc_model(alpha, delta1, delta2, r, u):
    N1, N2 = r.shape
    V1, V2 = list(six.moves.map(lambda x: 2 * x + 1, u))
    return (
        lambda n1, n2: alpha
        / (N1 * N2)
        * sin((n1 + delta1) * V1 / N1 * pi)
        * sin((n2 + delta2) * V2 / N2 * pi)
        / (sin((n1 + delta1) * pi / N1) * sin((n2 + delta2) * pi / N2))
    )


def pocfunc(f, g, windowfunc=np.hanning, withlpf=False):
    m = np.floor(list(six.moves.map(lambda x: x / 2.0, f.shape)))
    u = list(six.moves.map(lambda x: x / 2.0, m))

    # hanning window
    hy = windowfunc(f.shape[0])
    hx = windowfunc(f.shape[1])
    hw = hy.reshape(hy.shape[0], 1) * hx
    f = f * hw
    g = g * hw

    # compute 2d fft
    F = scipy.fftpack.fft2(f)
    G = scipy.fftpack.fft2(g)
    G_ = np.conj(G)
    R = F * G_ / np.abs(F * G_)

    if withlpf == True:
        R = scipy.fftpack.fftshift(R)
        lpf = np.ones(list(six.moves.map(lambda x: x + 1.0, m)))
        lpf = zero_padding(lpf, f.shape, u)
        R = R * lpf
        R = scipy.fftpack.fftshift(R)

    return scipy.fftpack.fftshift(np.real(scipy.fftpack.ifft2(R)))


def poc(f, g, fitting_shape=(9, 9)):
    # compute phase-only correlation
    center = list(six.moves.map(lambda x: x / 2.0, f.shape))
    m = np.floor(list(six.moves.map(lambda x: x / 2.0, f.shape)))
    u = list(six.moves.map(lambda x: x / 2.0, m))

    r = pocfunc(f, g)

    # least-square fitting
    max_pos = np.argmax(r)
    peak = (max_pos // f.shape[1], max_pos % f.shape[1])
    max_peak = r[peak[0], peak[1]]

    mf = list(six.moves.map(lambda x: int(x / 2), fitting_shape))
    fitting_area = r[
        peak[0] - mf[0] : peak[0] + mf[0] + 1, peak[1] - mf[1] : peak[1] + mf[1] + 1
    ]

    p0 = [0.5, -(peak[0] - m[0]) - 0.02, -(peak[1] - m[1]) - 0.02]
    y, x = np.mgrid[-mf[0] : mf[0] + 1, -mf[1] : mf[1] + 1]
    y = y + peak[0] - m[0]
    x = x + peak[1] - m[1]
    errorfunction = lambda p: np.ravel(
        pocfunc_model(p[0], p[1], p[2], r, u)(y, x) - fitting_area
    )
    plsq = leastsq(errorfunction, p0)
    return (plsq[0][0], plsq[0][1], plsq[0][2])


def ripoc(f, g, M=50, fitting_shape=(9, 9)):

    hy = np.hanning(f.shape[0])
    hx = np.hanning(f.shape[1])
    hw = hy.reshape(hy.shape[0], 1) * hx

    ff = f * hw
    gg = g * hw

    F = scipy.fftpack.fft2(ff)
    G = scipy.fftpack.fft2(gg)

    F = scipy.fftpack.fftshift(np.log(np.abs(F)))
    G = scipy.fftpack.fftshift(np.log(np.abs(G)))

    FLP = logpolar(F, (F.shape[0] / 2, F.shape[1] / 2), M)
    GLP = logpolar(G, (G.shape[0] / 2, G.shape[1] / 2), M)

    R = poc(FLP, GLP)

    angle = -R[1] / F.shape[0] * 360
    scale = 1.0 - R[2] / 100

    center = tuple(np.array(g.shape) / 2)
    rot = cv2.getRotationMatrix2D(center, -angle, 1.0 + (1.0 - scale))

    g_dash = cv2.warpAffine(g, rot, (g.shape[1], g.shape[0]), flags=cv2.INTER_LANCZOS4)

    t = poc(f, g_dash)

    return (t[1], t[2], angle, scale)

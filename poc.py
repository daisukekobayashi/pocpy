#! /usr/bin/env python
# -*- coding: utf-8
import sys

import numpy
from numpy import pi, sin, cos
from scipy.optimize import leastsq
import scipy, scipy.fftpack

import cv2
import cv2.cv as cv

def logpolar(src, center, magnitude_scale = 40):

    mat1 = cv.fromarray(numpy.float64(src))
    mat2 = cv.CreateMat(src.shape[0], src.shape[1], mat1.type)

    cv.LogPolar(mat1, mat2, center, magnitude_scale)

    return numpy.asarray(mat2)

def zero_padding(src, dstshape, pos = (0, 0)):
    y, x = pos
    dst = numpy.zeros(dstshape)
    dst[y:src.shape[0] + y, x:src.shape[1] + x] = src
    return dst

def pocfunc_model(alpha, delta1, delta2, r, u):
    N1, N2 = r.shape
    V1, V2 = map(lambda x: 2 * x + 1, u)
    return lambda n1, n2: alpha / (N1 * N2) * sin((n1 + delta1) * V1 / N1 * pi) * sin((n2 + delta2) * V2 / N2 * pi)\
                                            / (sin((n1 + delta1) * pi / N1) * sin((n2 + delta2) * pi / N2))

def pocfunc(f, g, windowfunc = numpy.hanning, withlpf = True):
    m = numpy.floor(map(lambda x: x / 2.0, f.shape))
    u = map(lambda x: x / 2.0, m)

    # hanning window
    hy = windowfunc(f.shape[0])
    hx = windowfunc(f.shape[1])
    hw = hy.reshape(hy.shape[0], 1) * hx
    f = f * hw
    g = g * hw

    # compute 2d fft
    F = scipy.fftpack.fft2(f)
    G = scipy.fftpack.fft2(g)
    G_ = numpy.conj(G)
    R = F * G_ / numpy.abs(F * G_)

    if withlpf == True:
        R = scipy.fftpack.fftshift(R)
        lpf = numpy.ones(map(lambda x: x + 1.0, m))
        lpf = zero_padding(lpf, f.shape, u)
        R = R * lpf
        R = scipy.fftpack.fftshift(R)

    return scipy.fftpack.fftshift(numpy.real(scipy.fftpack.ifft2(R)))

def poc(f, g, fitting_shape = (9, 9)):
    # compute phase-only correlation
    center = map(lambda x: x / 2.0, f.shape)
    m = numpy.floor(map(lambda x: x / 2.0, f.shape))
    u = map(lambda x: x / 2.0, m)

    r = pocfunc(f, g)

    print r

    # least-square fitting
    max_pos = numpy.argmax(r)
    peak = (max_pos / f.shape[1], max_pos % f.shape[1])
    max_peak = r[peak[0], peak[1]]

    print peak

    mf = numpy.floor(map(lambda x: x / 2.0, fitting_shape))
    fitting_area = r[peak[0] - mf[0] : peak[0] + mf[0] + 1,\
                     peak[1] - mf[1] : peak[1] + mf[1] + 1]

    p0 = [0.5, -(peak[0] - m[0]) - 0.02, -(peak[1] - m[1]) - 0.02]
    y, x = numpy.mgrid[-mf[0]:mf[0] + 1, -mf[1]:mf[1] + 1]
    y = y + peak[0] - m[0]
    x = x + peak[1] - m[1]
    errorfunction = lambda p: numpy.ravel(pocfunc_model(p[0], p[1], p[2], r, u)(y, x) - fitting_area)
    plsq = leastsq(errorfunction, p0)
    return (plsq[0][0], plsq[0][1], plsq[0][2])


if __name__ == '__main__':

    img1 = cv2.imread(sys.argv[1], 0)
    img2 = cv2.imread(sys.argv[2], 0)

    print poc(img1, img2)


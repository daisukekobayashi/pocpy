# pocpy (Phase Only Correlation in Python)

This repository contains implementations of phase only correlation and rotation invariant phase only correlation in python.

This is an unofficial implementation of the paper below. `poc` follows
Sections 2-3 and `ripoc` follows Section 5.

> K. Takita, T. Aoki, Y. Sasaki, T. Higuchi, and K. Kobayashi,
> "High-accuracy subpixel image registration based on phase-only
> correlation," IEICE Trans. Fundamentals, vol. E86-A, no. 8,
> pp. 1925-1934, Aug. 2003.

## Installing pocpy

```console
$ pip install pocpy
```

## Usage

```python
import cv2
import numpy as np

import pocpy

f = cv2.imread("reference.png", cv2.IMREAD_GRAYSCALE).astype(np.float64)
g = cv2.imread("target.png", cv2.IMREAD_GRAYSCALE).astype(np.float64)

# translation only
peak, dy, dx = pocpy.poc(f, g)

# translation, rotation and scale
peak, dy, dx, angle, scale = pocpy.ripoc(f, g)
```

- `f` and `g` are 2D arrays of the same shape.
- `poc` returns the sub-pixel translation `(dy, dx)` of `g` relative to `f`.
- `ripoc` assumes `g` is `f` translated by `(dy, dx)` and then rotated by
  `angle` degrees (counter-clockwise, as in `cv2.getRotationMatrix2D`) and
  scaled by `scale` about the image center.
- `peak` is the height of the correlation peak and can be used as a
  similarity measure.
- `angle` is between -180 and 180 degrees.

## Development

```console
$ uv sync
$ uv run pytest
$ uv run ruff check .
$ uv run ruff format --check .
```

import importlib

import numpy as np
import pytest
from conftest import crop, rotate_scale, shift

import pocpy


@pytest.mark.parametrize(
    "angle", [0, 10, -15, 30, 95, 120, 170, -100, -150, 179]
)
def test_ripoc_estimates_rotation(texture, angle):
    _, dy, dx, est_angle, scale = pocpy.ripoc(
        crop(texture), crop(rotate_scale(texture, angle, 1.0))
    )
    assert est_angle == pytest.approx(angle, abs=0.2)
    assert scale == pytest.approx(1.0, abs=0.01)
    assert dy == pytest.approx(0, abs=0.2)
    assert dx == pytest.approx(0, abs=0.2)


@pytest.mark.parametrize(
    ("angle", "dy", "dx"), [(10, 3, -5), (-20, -6, 4), (120, 3, -5)]
)
def test_ripoc_estimates_translation_before_rotation(texture, angle, dy, dx):
    # g is f translated by (dy, dx) and then rotated about the center.
    g = rotate_scale(shift(texture, dy, dx), angle, 1.0)
    _, est_dy, est_dx, est_angle, _ = pocpy.ripoc(crop(texture), crop(g))
    assert est_angle == pytest.approx(angle, abs=0.2)
    assert est_dy == pytest.approx(dy, abs=0.2)
    assert est_dx == pytest.approx(dx, abs=0.2)


@pytest.mark.parametrize(
    ("angle", "scale"),
    [(0, 1.1), (0, 0.9), (12, 0.9), (-8, 1.05), (-89, 0.9), (-135, 1.1)],
)
def test_ripoc_estimates_scale(texture, angle, scale):
    _, _, _, est_angle, est_scale = pocpy.ripoc(
        crop(texture), crop(rotate_scale(texture, angle, scale))
    )
    assert est_angle == pytest.approx(angle, abs=0.2)
    assert est_scale == pytest.approx(scale, abs=0.01)


def test_ripoc_estimates_all_parameters(texture):
    g = rotate_scale(shift(texture, -4, 2), -8, 1.05)
    peak, dy, dx, angle, scale = pocpy.ripoc(crop(texture), crop(g))
    assert angle == pytest.approx(-8, abs=0.2)
    assert scale == pytest.approx(1.05, abs=0.01)
    assert dy == pytest.approx(-4, abs=0.2)
    assert dx == pytest.approx(2, abs=0.2)
    assert peak > 0.3


def test_ripoc_passes_fitting_shape_to_poc(texture, monkeypatch):
    # `pocpy.poc` is the function re-exported by the package.
    module = importlib.import_module("pocpy.poc")
    calls = []
    original = module.poc

    def spy(*args, **kwargs):
        calls.append(kwargs.get("fitting_shape"))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "poc", spy)
    pocpy.ripoc(crop(texture), crop(texture), fitting_shape=(7, 7))
    assert calls and all(c == (7, 7) for c in calls)


@pytest.mark.parametrize(
    ("height", "width"), [(256, 384), (384, 256), (200, 300), (384, 384)]
)
def test_ripoc_supports_any_image_size(texture, height, width):
    g = rotate_scale(shift(texture, -4, 2), -8, 1.05)
    _, dy, dx, angle, scale = pocpy.ripoc(
        crop(texture, height, width), crop(g, height, width)
    )
    assert angle == pytest.approx(-8, abs=0.2)
    assert scale == pytest.approx(1.05, abs=0.01)
    assert dy == pytest.approx(-4, abs=0.2)
    assert dx == pytest.approx(2, abs=0.2)


def test_ripoc_uses_given_magnitude_scale(texture):
    g = crop(rotate_scale(texture, 0, 1.1))
    *_, auto = pocpy.ripoc(crop(texture), g)
    *_, manual = pocpy.ripoc(crop(texture), g, M=50)
    assert auto == pytest.approx(1.1, abs=0.01)
    assert manual == pytest.approx(1.1, abs=0.01)
    assert auto != manual


def test_ripoc_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        pocpy.ripoc(np.ones((256, 256)), np.ones((128, 128)))

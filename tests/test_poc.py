import numpy as np
import pytest
from conftest import crop, shift

import pocpy


def test_pocfunc_peak_is_at_center_for_identical_images(texture):
    f = crop(texture)
    r = pocpy.pocfunc(f, f)
    assert r.shape == f.shape
    assert np.unravel_index(np.argmax(r), r.shape) == (128, 128)


@pytest.mark.parametrize(
    ("dy", "dx"), [(0, 0), (5, -12), (-20, 7), (3.3, -7.6), (-0.4, 0.7)]
)
def test_poc_estimates_translation(texture, dy, dx):
    peak, est_dy, est_dx = pocpy.poc(
        crop(texture), crop(shift(texture, dy, dx))
    )
    assert est_dy == pytest.approx(dy, abs=0.1)
    assert est_dx == pytest.approx(dx, abs=0.1)
    assert peak > 0.5


def test_poc_estimates_translation_with_lpf(texture):
    _, dy, dx = pocpy.poc(
        crop(texture), crop(shift(texture, 4.5, -2.5)), withlpf=True
    )
    assert dy == pytest.approx(4.5, abs=0.1)
    assert dx == pytest.approx(-2.5, abs=0.1)


def test_poc_peak_is_lower_for_unrelated_images(texture):
    other = np.random.default_rng(1).random(texture.shape)
    same, *_ = pocpy.poc(crop(texture), crop(shift(texture, 2, 3)))
    diff, *_ = pocpy.poc(crop(texture), crop(other))
    assert diff < 0.2 < same


def test_poc_non_square(texture):
    f = texture[100:300, 50:350]
    g = shift(texture, 4, -6)[100:300, 50:350]
    _, dy, dx = pocpy.poc(f, g)
    assert dy == pytest.approx(4, abs=0.1)
    assert dx == pytest.approx(-6, abs=0.1)


@pytest.mark.parametrize("dy", [124, 126, 127])
def test_poc_peak_near_border(texture, dy):
    # The overlap is tiny, so only check the fit stays near the peak.
    g = np.roll(texture, dy, axis=0)
    _, est_dy, _ = pocpy.poc(crop(texture), crop(g))
    assert est_dy == pytest.approx(dy, abs=1)

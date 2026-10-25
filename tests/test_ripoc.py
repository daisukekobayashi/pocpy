import cv2
import pytest
from conftest import crop, rotate_scale, shift

import pocpy

pytestmark = pytest.mark.xfail(
    not hasattr(cv2, "logPolar"),
    raises=AttributeError,
    strict=True,
    reason="cv2.logPolar was removed in OpenCV 5",
)

scale_bug = pytest.mark.xfail(
    strict=True,
    reason="scale is computed as 1 - d / 100 instead of exp(d / M)",
)


@pytest.mark.parametrize("angle", [0, 10, -15, 30])
def test_ripoc_estimates_rotation(texture, angle):
    _, dy, dx, est_angle, scale = pocpy.ripoc(
        crop(texture), crop(rotate_scale(texture, angle, 1.0))
    )
    assert est_angle == pytest.approx(angle, abs=0.2)
    assert scale == pytest.approx(1.0, abs=0.01)
    assert dy == pytest.approx(0, abs=0.2)
    assert dx == pytest.approx(0, abs=0.2)


@pytest.mark.parametrize(("angle", "dy", "dx"), [(10, 3, -5), (-20, -6, 4)])
def test_ripoc_estimates_translation_before_rotation(texture, angle, dy, dx):
    # g is f translated by (dy, dx) and then rotated about the center.
    g = rotate_scale(shift(texture, dy, dx), angle, 1.0)
    _, est_dy, est_dx, est_angle, _ = pocpy.ripoc(crop(texture), crop(g))
    assert est_angle == pytest.approx(angle, abs=0.2)
    assert est_dy == pytest.approx(dy, abs=0.2)
    assert est_dx == pytest.approx(dx, abs=0.2)


@scale_bug
@pytest.mark.parametrize(("angle", "scale"), [(0, 1.1), (0, 0.9), (12, 0.9)])
def test_ripoc_estimates_scale(texture, angle, scale):
    _, _, _, est_angle, est_scale = pocpy.ripoc(
        crop(texture), crop(rotate_scale(texture, angle, scale))
    )
    assert est_angle == pytest.approx(angle, abs=0.2)
    assert est_scale == pytest.approx(scale, abs=0.01)

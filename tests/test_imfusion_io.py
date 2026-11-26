import numpy as np
import pytest

from imfusion_io import from_shared_image, to_shared_image


def test_VS011_shared_image_preserves_axis_order_and_spacing() -> None:
    imfusion = pytest.importorskip("imfusion", exc_type=ImportError)
    volume = np.zeros((3, 5, 7), dtype=np.float32)
    volume[1, 2, 4] = 123.0
    spacing_xyz_mm = (0.4, 0.7, 1.2)

    try:
        imfusion.init(initOpenGL=False)
        image = to_shared_image(volume, spacing_xyz_mm)
        restored, restored_spacing = from_shared_image(image)
    except imfusion.MissingLicenseError:
        pytest.skip("ImFusion license unavailable")

    assert (image.slices, image.height, image.width) == (3, 5, 7)
    assert (image.spacing.x, image.spacing.y, image.spacing.z) == pytest.approx(spacing_xyz_mm)
    assert restored.shape == (3, 5, 7)
    assert restored.dtype == np.float32
    assert restored[1, 2, 4] == 123.0
    np.testing.assert_array_equal(restored, volume)
    assert restored_spacing == pytest.approx(spacing_xyz_mm)

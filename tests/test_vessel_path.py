import numpy as np
import pytest
from scipy.spatial import cKDTree

from vessel_path import extract_centerline


def test_RD004_centerline_stays_within_one_voxel_of_curved_tube_axis():
    x = np.linspace(4, 48, 881)
    true_centerline = np.column_stack((np.full_like(x, 11), 12 + 6 * np.sin((x - 6) * np.pi / 40), x))
    tree = cKDTree(true_centerline)
    shape = (23, 25, 53)
    grid = np.indices(shape).reshape(3, -1).T
    mask = tree.query(grid)[0].reshape(shape) <= 4.5
    start, end = (11, 12, 6), (11, 12, 46)

    path = extract_centerline(mask, start, end, spacing_mm=0.4)

    np.testing.assert_array_equal(path[[0, -1]], [start, end])
    assert mask[tuple(path.T)].all()
    assert np.max(np.abs(np.diff(path, axis=0))) == 1
    deviation = tree.query(path)[0].max()
    assert deviation <= 1, f"Maximum deviation was {deviation:.3f} voxels"


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        ((-1, 2, 1), (2, 2, 3), "Start is outside"),
        ((2, 1, 1), (2, 2, 3), "Start is outside"),
        ((2, 2, 1), (2, 2, 5), "End is outside"),
        ((2, 2, 1), (2, 1, 3), "End is outside"),
        ((2.5, 2, 1), (2, 2, 3), "Start must contain three integer"),
    ],
)
def test_RD005_rejects_invalid_endpoints(start, end, message):
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[2, 2, 1:4] = True

    with pytest.raises(ValueError, match=message):
        extract_centerline(mask, start, end, spacing_mm=0.4)


def test_RD005_rejects_disconnected_endpoints():
    mask = np.zeros((8, 8, 8), dtype=bool)
    mask[1:3, 1:3, 1:3] = True
    mask[4:6, 4:6, 4:6] = True

    with pytest.raises(ValueError, match="No connected vessel path"):
        extract_centerline(mask, (2, 2, 2), (4, 4, 4), spacing_mm=0.4)


@pytest.mark.parametrize("spacing", [0, -0.4, np.nan, np.inf])
def test_RD005_rejects_invalid_spacing(spacing):
    mask = np.ones((3, 3, 3), dtype=bool)

    with pytest.raises(ValueError, match="spacing must be finite and positive"):
        extract_centerline(mask, (1, 1, 0), (1, 1, 2), spacing_mm=spacing)

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from imfusion import SharedImage


def to_shared_image(
    volume: NDArray[np.float32], spacing_xyz_mm: tuple[float, float, float]
) -> SharedImage:
    from imfusion import SharedImage

    if volume.ndim != 3 or volume.size == 0:
        raise ValueError("Provide a nonempty volume in (z, y, x) order.")
    spacing = np.asarray(spacing_xyz_mm, dtype=np.float64)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing) & (spacing > 0)):
        raise ValueError("Provide three finite, positive spacing values in (x, y, z) order.")

    image = SharedImage(np.ascontiguousarray(volume, dtype=np.float32), greyscale=True)
    image.spacing = spacing
    return image


def from_shared_image(
    image: SharedImage,
) -> tuple[NDArray[np.float32], tuple[float, float, float]]:
    if image.channels != 1:
        raise ValueError("Provide a single-channel SharedImage.")

    # ImFusion stores arrays in (z, y, x, channels) and spacing in (x, y, z).
    volume = np.array(image, dtype=np.float32).reshape(image.slices, image.height, image.width)
    spacing = image.spacing
    return volume, (spacing.x, spacing.y, spacing.z)

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class StenosisResult:
    reference_diameter_mm: float
    min_diameter_mm: float
    min_slice_index: int
    percent_stenosis: float


def measure_stenosis(
    volume: NDArray[np.number],
    spacing_mm: float,
    threshold: float | None = None,
) -> StenosisResult:
    if volume.ndim != 3:
        raise ValueError("volume must be three-dimensional.")
    if not np.isfinite(spacing_mm) or spacing_mm <= 0:
        raise ValueError("spacing_mm must be finite and positive.")
    reference_slice_count = 10
    if volume.shape[0] < 2 * reference_slice_count:
        raise ValueError("volume must have at least 20 slices.")
    if volume.size == 0:
        raise ValueError("No lumen found in the volume.")

    if threshold is None:
        threshold = (float(volume.min()) + float(volume.max())) / 2
    voxel_counts = np.count_nonzero(volume > threshold, axis=(1, 2))
    if not np.any(voxel_counts):
        raise ValueError("No lumen found in the volume.")

    diameters = 2 * np.sqrt(voxel_counts / np.pi) * spacing_mm
    reference_diameters = np.concatenate(
        (diameters[:reference_slice_count], diameters[-reference_slice_count:])
    )
    reference_diameter_mm = float(np.median(reference_diameters))
    if reference_diameter_mm == 0:
        raise ValueError("No lumen found in the median reference slice.")

    narrowest_slices = np.flatnonzero(voxel_counts == voxel_counts.min())
    min_slice_index = int(narrowest_slices[len(narrowest_slices) // 2])
    min_diameter_mm = float(diameters[min_slice_index])
    return StenosisResult(
        reference_diameter_mm=reference_diameter_mm,
        min_diameter_mm=min_diameter_mm,
        min_slice_index=min_slice_index,
        percent_stenosis=(1 - min_diameter_mm / reference_diameter_mm) * 100,
    )

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import label

from centerline import sample_cross_sections


@dataclass(frozen=True)
class CurvedStenosisResult:
    distance_mm: NDArray[np.float64]
    diameter_mm: NDArray[np.float64]
    reference_diameter_mm: float
    min_diameter_mm: float
    min_distance_mm: float
    percent_stenosis: float


def measure_curved_stenosis(
    volume: NDArray[np.number],
    centerline: NDArray[np.number],
    spacing_mm: float,
    threshold: float | None = None,
    *,
    step_mm: float | None = None,
    section_size: int = 62,
    pixel_spacing_mm: float | None = None,
    require_center_pixel: bool = False,
) -> CurvedStenosisResult:
    points = np.asarray(centerline)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 10:
        raise ValueError("centerline must have shape (N, 3) with at least 10 points.")

    if step_mm is None:
        step_mm = spacing_mm
    if pixel_spacing_mm is None:
        pixel_spacing_mm = spacing_mm / 2
    if require_center_pixel and section_size % 2 == 0:
        raise ValueError("An odd section size is required to sample the center pixel.")
    sections, distance_mm = sample_cross_sections(
        volume, points, spacing_mm, step_mm, section_size, pixel_spacing_mm
    )
    if threshold is None:
        threshold = (float(volume.min()) + float(volume.max())) / 2
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")

    center = (section_size - 1) / 2
    offsets_mm = (np.arange(section_size) - center) * pixel_spacing_mm
    center_distance_mm = np.hypot(offsets_mm[:, None], offsets_mm[None, :])
    near_center = center_distance_mm <= (0 if require_center_pixel else spacing_mm)
    pixel_counts = np.array([
        _lumen_pixel_count(section > threshold, near_center, center_distance_mm)
        for section in sections
    ])
    areas = pixel_counts * pixel_spacing_mm**2
    if not np.any(areas):
        raise ValueError("No lumen found along the centerline.")

    diameter_mm = 2 * np.sqrt(areas / np.pi)
    path_length_mm = distance_mm[-1]
    reference = (distance_mm <= 0.15 * path_length_mm) | (distance_mm >= 0.85 * path_length_mm)
    reference_diameter_mm = float(np.median(diameter_mm[reference]))
    if reference_diameter_mm == 0:
        raise ValueError("No lumen found in the median reference cross-section.")

    minimum_indices = np.flatnonzero(pixel_counts == pixel_counts.min())
    minimum_index = minimum_indices[len(minimum_indices) // 2]
    min_diameter_mm = float(diameter_mm[minimum_index])
    return CurvedStenosisResult(
        distance_mm=distance_mm,
        diameter_mm=diameter_mm,
        reference_diameter_mm=reference_diameter_mm,
        min_diameter_mm=min_diameter_mm,
        min_distance_mm=float(distance_mm[minimum_index]),
        percent_stenosis=(1 - min_diameter_mm / reference_diameter_mm) * 100,
    )


def _lumen_pixel_count(
    lumen: NDArray[np.bool_],
    near_center: NDArray[np.bool_],
    center_distance_mm: NDArray[np.float64],
) -> int:
    candidates = lumen & near_center
    if not np.any(candidates):
        return 0
    components, _ = label(lumen)
    nearest = np.argmin(center_distance_mm[candidates])
    component = components[candidates][nearest]
    return int(np.count_nonzero(components == component))

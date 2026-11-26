import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d, map_coordinates


def resample_centerline(
    centerline: NDArray[np.number],
    spacing_mm: float,
    step_mm: float = 0.5,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    points = np.asarray(centerline, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError("centerline must have shape (N, 3) with at least two points.")
    if not np.all(np.isfinite(points)):
        raise ValueError("centerline must contain finite coordinates.")
    if not np.isfinite(spacing_mm) or spacing_mm <= 0:
        raise ValueError("spacing_mm must be finite and positive.")
    if not np.isfinite(step_mm) or step_mm <= 0:
        raise ValueError("step_mm must be finite and positive.")

    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    distance = np.concatenate(([0.0], np.cumsum(segments)))
    if distance[-1] == 0:
        raise ValueError("centerline must have positive length.")

    distinct = np.concatenate(([True], np.diff(distance) > 0))
    intervals = max(1, int(np.ceil(distance[-1] / (step_mm / spacing_mm))))
    sampled_distance = np.linspace(0, distance[-1], intervals + 1)
    sampled = np.column_stack(
        [np.interp(sampled_distance, distance[distinct], axis[distinct]) for axis in points.T]
    )
    return sampled, sampled_distance * spacing_mm


def estimate_tangents(
    points: NDArray[np.float64], smoothing_sigma: float = 2.0
) -> NDArray[np.float64]:
    if not np.isfinite(smoothing_sigma) or smoothing_sigma < 0:
        raise ValueError("smoothing_sigma must be finite and nonnegative.")
    derivatives = np.gradient(np.asarray(points, dtype=np.float64), axis=0)
    if smoothing_sigma > 0:
        derivatives = gaussian_filter1d(derivatives, smoothing_sigma, axis=0, mode="nearest")
    lengths = np.linalg.norm(derivatives, axis=1)
    if np.any(lengths == 0):
        raise ValueError("Cannot estimate a tangent at every centerline point.")
    return derivatives / lengths[:, None]


def parallel_transport_frames(
    tangents: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    normals = np.empty_like(tangents)
    axis = np.eye(3)[np.argmin(np.abs(tangents[0]))]
    normal = axis - np.dot(axis, tangents[0]) * tangents[0]
    normals[0] = normal / np.linalg.norm(normal)
    for index, (previous, current) in enumerate(zip(tangents[:-1], tangents[1:]), start=1):
        rotation = np.cross(previous, current)
        sine = np.linalg.norm(rotation)
        normal = normals[index - 1]
        if sine > 1e-12:
            axis = rotation / sine
            cosine = np.dot(previous, current)
            normal = (
                normal * cosine
                + np.cross(axis, normal) * sine
                + axis * np.dot(axis, normal) * (1 - cosine)
            )
        normal = normal - np.dot(normal, current) * current
        normals[index] = normal / np.linalg.norm(normal)
    return normals, np.cross(tangents, normals)


def sample_cross_sections(
    volume: NDArray[np.number],
    centerline: NDArray[np.number],
    spacing_mm: float,
    step_mm: float = 0.5,
    section_size: int = 41,
    pixel_spacing_mm: float = 0.25,
    smoothing_sigma: float = 2.0,
) -> tuple[NDArray[np.float32], NDArray[np.float64]]:
    if volume.ndim != 3 or volume.size == 0:
        raise ValueError("volume must be a nonempty three-dimensional array.")
    if not isinstance(section_size, (int, np.integer)) or section_size <= 0:
        raise ValueError("section_size must be a positive integer.")
    if not np.isfinite(pixel_spacing_mm) or pixel_spacing_mm <= 0:
        raise ValueError("pixel_spacing_mm must be finite and positive.")

    points, distance_mm = resample_centerline(centerline, spacing_mm, step_mm)
    if np.any(np.asarray(centerline) < 0) or np.any(centerline > np.array(volume.shape) - 1):
        raise ValueError("centerline must lie inside the volume.")
    normals, binormals = parallel_transport_frames(estimate_tangents(points, smoothing_sigma))
    offsets = (np.arange(section_size) - (section_size - 1) / 2) * pixel_spacing_mm / spacing_mm

    # Coordinates stay in (z, y, x); rows follow the normal and columns the binormal.
    coordinates = (
        points[:, None, None, :]
        + offsets[None, :, None, None] * normals[:, None, None, :]
        + offsets[None, None, :, None] * binormals[:, None, None, :]
    )
    sections = map_coordinates(
        volume,
        np.moveaxis(coordinates, -1, 0),
        output=np.float32,
        order=1,
        cval=float(volume.min()),
    )
    return sections, distance_mm

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from centerline import (
    estimate_tangents,
    parallel_transport_frames,
    resample_centerline,
    sample_cross_sections,
)
from curved_vessel_phantom import CurvedVesselSpec, true_centerline


def test_VC004_resampling_preserves_arc_length_and_equal_spacing() -> None:
    spec = CurvedVesselSpec()
    centerline = true_centerline(spec)
    indices = np.r_[np.arange(0, len(centerline) - 1, 11), len(centerline) - 1]
    points, distance_mm = resample_centerline(centerline[indices], spec.spacing_mm)
    analytic_length_mm = spec.bend_radius * np.deg2rad(spec.sweep_deg) * spec.spacing_mm
    expected_step_mm = distance_mm[-1] / (len(points) - 1)
    chord_lengths_mm = np.linalg.norm(np.diff(points, axis=0), axis=1) * spec.spacing_mm

    np.testing.assert_array_equal(points[[0, -1]], centerline[[0, -1]])
    np.testing.assert_allclose(chord_lengths_mm, expected_step_mm, rtol=0.01)
    np.testing.assert_allclose(np.diff(distance_mm), expected_step_mm)
    assert distance_mm[0] == 0
    assert distance_mm[-1] == pytest.approx(analytic_length_mm, rel=0.01)


def test_VC005_frames_are_orthonormal_and_parallel_transported() -> None:
    theta = np.linspace(0, 2 * np.pi, 250)
    helix = np.column_stack((theta, 3 * np.sin(theta), 3 * np.cos(theta)))
    tangents = estimate_tangents(helix)
    normals, binormals = parallel_transport_frames(tangents)
    frames = np.stack((tangents, normals, binormals), axis=1)
    analytic_tangents = np.column_stack(
        (np.ones_like(theta), 3 * np.cos(theta), -3 * np.sin(theta))
    ) / np.sqrt(10)

    np.testing.assert_allclose(tangents[10:-10], analytic_tangents[10:-10], atol=0.001)
    np.testing.assert_allclose(
        frames @ frames.transpose(0, 2, 1),
        np.broadcast_to(np.eye(3), frames.shape),
        atol=1e-12,
    )
    np.testing.assert_allclose(np.cross(tangents, normals), binormals, atol=1e-12)

    axes = np.cross(tangents[:-1], tangents[1:])
    sines = np.linalg.norm(axes, axis=1)
    cosines = np.sum(tangents[:-1] * tangents[1:], axis=1)
    rotations = Rotation.from_rotvec(axes * (np.arctan2(sines, cosines) / sines)[:, None])
    np.testing.assert_allclose(rotations.apply(normals[:-1]), normals[1:], atol=1e-12)
    np.testing.assert_allclose(rotations.apply(binormals[:-1]), binormals[1:], atol=1e-12)


def test_VC005_sections_sample_the_perpendicular_plane() -> None:
    z, y, x = np.indices((20, 41, 41), dtype=np.float32)
    volume = 100 * z + 10 * y + x
    centerline = np.array([[4, 20, 20], [11, 20, 20]], dtype=float)
    sections, distance_mm = sample_cross_sections(volume, centerline, spacing_mm=0.5)
    offsets = np.arange(-20, 21) * 0.5
    expected = (
        100 * (4 + distance_mm[:, None, None] / 0.5)
        + 10 * (20 + offsets[None, :, None])
        + 20 + offsets[None, None, :]
    )

    assert sections.shape == (8, 41, 41)
    np.testing.assert_allclose(sections, expected, atol=1e-6, rtol=0)


def test_VC010_spacing_scales_centerline_and_sampling_lengths() -> None:
    centerline = true_centerline()
    points, distance_mm = resample_centerline(centerline, spacing_mm=0.5, step_mm=0.5)
    scaled_points, scaled_distance_mm = resample_centerline(
        centerline, spacing_mm=1.5, step_mm=1.5
    )
    np.testing.assert_allclose(scaled_points, points)
    np.testing.assert_allclose(scaled_distance_mm, 3 * distance_mm)

    volume = np.arange(20**3, dtype=np.float32).reshape((20,) * 3)
    straight = np.array([[2, 10, 9], [11, 10, 9]], dtype=float)
    sections, distance_mm = sample_cross_sections(volume, straight, 0.5, section_size=9)
    scaled_sections, scaled_distance_mm = sample_cross_sections(
        volume, straight, 1.5, step_mm=1.5, section_size=9, pixel_spacing_mm=0.75
    )
    np.testing.assert_array_equal(scaled_sections, sections)
    np.testing.assert_allclose(scaled_distance_mm, 3 * distance_mm)

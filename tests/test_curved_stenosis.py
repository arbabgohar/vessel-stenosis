import numpy as np
import pytest

from curved_stenosis import measure_curved_stenosis
from curved_vessel_phantom import CurvedVesselSpec, generate_curved_phantom, true_centerline
from stenosis import measure_stenosis
from vessel_phantom import VesselSpec, generate_phantom


@pytest.mark.parametrize("min_radius", [9.0, 6.0, 3.0])
def test_VC006_matches_straight_vessel_measurement(min_radius: float) -> None:
    spec = VesselSpec(min_radius=min_radius)
    volume = generate_phantom(spec)
    center = (spec.size_xy - 1) / 2
    centerline = np.column_stack(
        (np.arange(spec.length), np.full((spec.length, 2), center))
    )
    baseline = measure_stenosis(volume, spec.spacing_mm)
    result = measure_curved_stenosis(volume, centerline, spec.spacing_mm)
    assert result.percent_stenosis == pytest.approx(baseline.percent_stenosis, abs=1)


@pytest.mark.parametrize("min_radius", [6.0, 4.0, 2.0])
def test_VC007_accuracy_noise_free(min_radius: float) -> None:
    spec = CurvedVesselSpec(min_radius=min_radius)
    result = measure_curved_stenosis(
        generate_curved_phantom(spec), true_centerline(spec), spec.spacing_mm
    )
    error = result.percent_stenosis - spec.true_stenosis_percent
    print(f"VC-007 truth={spec.true_stenosis_percent:g}% error={error:+.3f} pp")
    assert abs(error) <= 5


def test_VC008_perpendicular_slices_reduce_error() -> None:
    spec = CurvedVesselSpec()
    volume = generate_curved_phantom(spec)
    centerline = true_centerline(spec)
    first_z, last_z = np.rint(centerline[[0, -1], 0]).astype(int)
    naive = measure_stenosis(volume[first_z:last_z + 1], spec.spacing_mm)
    perpendicular = measure_curved_stenosis(volume, centerline, spec.spacing_mm)
    naive_error = abs(naive.percent_stenosis - spec.true_stenosis_percent)
    perpendicular_error = abs(perpendicular.percent_stenosis - spec.true_stenosis_percent)
    print(
        f"VC-008 absolute errors: perpendicular={perpendicular_error:.3f} pp, "
        f"axis-aligned={naive_error:.3f} pp"
    )
    assert perpendicular_error < naive_error


@pytest.mark.parametrize("min_radius", [6.0, 4.0, 2.0])
@pytest.mark.parametrize(
    ("noise_sigma", "jitter_voxels"),
    [(100.0, 0.0), (0.0, 0.5), (100.0, 0.5)],
    ids=["noise", "jitter", "noise-and-jitter"],
)
def test_VC009_accuracy_with_noise_and_jitter(
    min_radius: float, noise_sigma: float, jitter_voxels: float
) -> None:
    spec = CurvedVesselSpec(min_radius=min_radius, noise_sigma=noise_sigma, seed=42)
    centerline = true_centerline(spec)
    if jitter_voxels:
        centerline = centerline[np.r_[np.arange(0, len(centerline) - 1, 10), len(centerline) - 1]]
        rng = np.random.default_rng(17)
        directions = rng.normal(size=centerline.shape)
        magnitudes = rng.uniform(0, jitter_voxels, len(centerline))
        centerline += directions / np.linalg.norm(directions, axis=1)[:, None] * magnitudes[:, None]
    result = measure_curved_stenosis(generate_curved_phantom(spec), centerline, spec.spacing_mm)
    error = result.percent_stenosis - spec.true_stenosis_percent
    print(
        f"VC-009 truth={spec.true_stenosis_percent:g}% noise={noise_sigma:g} "
        f"jitter={jitter_voxels:g} voxel error={error:+.3f} pp"
    )
    assert abs(error) <= 5


def test_VC010_spacing_scales_profile_without_changing_percent() -> None:
    spec = CurvedVesselSpec()
    volume = generate_curved_phantom(spec)
    centerline = true_centerline(spec)
    original = measure_curved_stenosis(volume, centerline, spec.spacing_mm)
    scaled = measure_curved_stenosis(volume, centerline, 3 * spec.spacing_mm)
    np.testing.assert_allclose(scaled.distance_mm, 3 * original.distance_mm)
    np.testing.assert_allclose(scaled.diameter_mm, 3 * original.diameter_mm)
    assert scaled.reference_diameter_mm == pytest.approx(3 * original.reference_diameter_mm)
    assert scaled.min_diameter_mm == pytest.approx(3 * original.min_diameter_mm)
    assert scaled.min_distance_mm == pytest.approx(3 * original.min_distance_mm)
    assert scaled.percent_stenosis == pytest.approx(original.percent_stenosis)

    straight = np.column_stack((np.arange(20), np.full((20, 2), 4)))
    straight_volume = np.zeros((20, 9, 9), dtype=np.float32)
    straight_volume[:, 2:7, 2:7] = 1000
    default = measure_curved_stenosis(straight_volume, straight, 0.5)
    rescaled = measure_curved_stenosis(straight_volume, straight, 0.7)
    np.testing.assert_allclose(rescaled.distance_mm, 1.4 * default.distance_mm)
    np.testing.assert_allclose(rescaled.diameter_mm, 1.4 * default.diameter_mm)


def test_profile_uses_nearest_lumen_and_ignores_distant_components() -> None:
    volume = np.zeros((20, 31, 31), dtype=np.float32)
    centerline = np.column_stack((np.arange(20), np.full((20, 2), 15)))
    volume[:, 16:19, 14:17] = 1000
    volume[:, 3:10, 3:10] = 1000
    volume[10, 16:19, 14:17] = 0

    result = measure_curved_stenosis(volume, centerline, 0.5)
    assert result.reference_diameter_mm == pytest.approx(2 * np.sqrt(9 * 0.5**2 / np.pi))
    assert result.diameter_mm[10] == result.min_diameter_mm == 0
    assert result.min_distance_mm == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("shape", "point_count", "center_x", "message"),
    [
        ((20, 8), 10, 4, "three-dimensional"),
        ((20, 8, 8), 9, 4, "at least 10 points"),
        ((20, 8, 8), 10, -0.1, "inside the volume"),
        ((20, 8, 8), 10, 7.1, "inside the volume"),
        ((20, 8, 8), 10, 4, "No lumen found along the centerline"),
    ],
    ids=["non-3d", "short-centerline", "below-volume", "above-volume", "no-lumen"],
)
def test_VC011_invalid_measurement_inputs(
    shape: tuple[int, ...], point_count: int, center_x: float, message: str
) -> None:
    volume = np.zeros(shape, dtype=np.float32)
    centerline = np.column_stack(
        (np.arange(point_count), np.full(point_count, 4), np.full(point_count, center_x))
    )
    with pytest.raises(ValueError, match=message):
        measure_curved_stenosis(volume, centerline, 0.5)

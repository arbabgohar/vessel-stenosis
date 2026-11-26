from dataclasses import replace

import numpy as np
import pytest

from curved_vessel_phantom import CurvedVesselSpec, generate_curved_phantom, true_centerline


def test_VC001_curved_geometry_and_ground_truth() -> None:
    spec = CurvedVesselSpec(size=161)
    centerline = true_centerline(spec)
    volume = generate_curved_phantom(spec)
    middle = (spec.size - 1) / 2
    half_sweep = np.deg2rad(spec.sweep_deg / 2)
    circle_x = middle - spec.bend_radius * (1 + np.cos(half_sweep)) / 2
    z, y, x = centerline.T

    assert centerline.ndim == 2 and centerline.shape[1] == 3
    np.testing.assert_allclose(y, middle)
    np.testing.assert_allclose(np.hypot(z - middle, x - circle_x), spec.bend_radius)
    angles = np.arctan2(z - middle, x - circle_x)
    np.testing.assert_allclose(angles[[0, -1]], [-half_sweep, half_sweep])
    assert volume.shape == (spec.size,) * 3
    assert volume.dtype == np.float32
    assert set(np.unique(volume)) == {spec.background_value, spec.lumen_value}
    assert spec.true_stenosis_percent == pytest.approx(50.0)

    for arc_distance, expected_radius in [
        (0, spec.min_radius),
        (spec.stenosis_length / 2, (spec.min_radius + spec.ref_radius) / 2),
        (spec.stenosis_length + 2, spec.ref_radius),
    ]:
        index = np.argmin(np.abs(angles * spec.bend_radius - arc_distance))
        z_index, y_index, x_index = np.rint(centerline[index]).astype(int)
        offsets = np.abs(np.arange(spec.size) - y_index)
        column = volume[z_index, :, x_index]
        assert np.all(column[offsets <= expected_radius - 1] == spec.lumen_value)
        assert np.all(column[offsets >= expected_radius + 1] == spec.background_value)


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("min_radius", 0.0),
        ("min_radius", -1.0),
        ("min_radius", 8.0),
        ("min_radius", 9.0),
        ("size", 105),
        ("stenosis_length", 0.0),
        ("stenosis_length", -1.0),
        ("stenosis_length", 48.0),
        ("spacing_mm", 0.0),
        ("spacing_mm", -0.5),
        ("noise_sigma", -1.0),
    ],
    ids=[
        "zero-min-radius",
        "negative-min-radius",
        "equal-radii",
        "min-radius-above-reference",
        "insufficient-margin",
        "zero-stenosis-length",
        "negative-stenosis-length",
        "stenosis-length-too-long",
        "zero-spacing",
        "negative-spacing",
        "negative-noise",
    ],
)
def test_VC002_invalid_curved_phantom_parameters(parameter: str, value: int | float) -> None:
    with pytest.raises(ValueError):
        CurvedVesselSpec(**{parameter: value})


def test_VC003_seeded_gaussian_noise_is_reproducible() -> None:
    spec = CurvedVesselSpec(size=110, noise_sigma=100.0, seed=42)
    first = generate_curved_phantom(spec)
    second = generate_curved_phantom(spec)
    different_seed = generate_curved_phantom(replace(spec, seed=43))
    clean = generate_curved_phantom(replace(spec, noise_sigma=0))

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different_seed)
    assert first.dtype == np.float32
    noise = first - clean
    assert noise.mean() == pytest.approx(0, abs=0.5)
    assert noise.std() == pytest.approx(spec.noise_sigma, rel=0.01)
    np.testing.assert_allclose(
        np.quantile(noise, [0.025, 0.5, 0.975]),
        [-1.96 * spec.noise_sigma, 0, 1.96 * spec.noise_sigma],
        atol=1,
    )

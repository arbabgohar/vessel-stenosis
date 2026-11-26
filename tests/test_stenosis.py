import numpy as np
import pytest

from stenosis import measure_stenosis
from vessel_phantom import VesselSpec, generate_phantom


def test_VS001_phantom_geometry() -> None:
    spec = VesselSpec()
    volume = generate_phantom(spec)
    center_xy = spec.size_xy // 2
    center_slice = spec.length // 2
    offsets = np.arange(spec.size_xy) - center_xy
    distances = np.hypot(offsets[:, None], offsets[None, :])
    lumen = volume == spec.lumen_value

    assert volume.shape == (spec.length, spec.size_xy, spec.size_xy)
    assert volume.dtype == np.float32
    assert lumen[:, center_xy, center_xy].all()
    assert np.all(volume[~lumen] == spec.background_value)
    np.testing.assert_array_equal(lumen[0], distances <= spec.ref_radius)
    np.testing.assert_array_equal(lumen[-1], distances <= spec.ref_radius)
    np.testing.assert_array_equal(lumen[center_slice], distances <= spec.min_radius)
    halfway_slice = center_slice + int(spec.stenosis_length / 2)
    np.testing.assert_array_equal(
        lumen[halfway_slice], distances <= (spec.ref_radius + spec.min_radius) / 2
    )
    np.testing.assert_array_equal(lumen, lumen[::-1])


@pytest.mark.parametrize(
    ("ref_radius", "min_radius", "expected_percent"),
    [(12.0, 9.0, 25.0), (10.0, 5.0, 50.0), (8.0, 2.0, 75.0)],
)
def test_VS002_ground_truth(
    ref_radius: float, min_radius: float, expected_percent: float
) -> None:
    spec = VesselSpec(ref_radius=ref_radius, min_radius=min_radius)
    assert spec.true_stenosis_percent == pytest.approx(expected_percent)


def test_VS003_reference_diameter_in_mm() -> None:
    spec = VesselSpec(spacing_mm=0.7)
    result = measure_stenosis(generate_phantom(spec), spec.spacing_mm)
    assert result.reference_diameter_mm == pytest.approx(
        2 * spec.ref_radius * spec.spacing_mm, rel=0.05
    )


def test_VS004_narrowest_slice() -> None:
    spec = VesselSpec()
    volume = generate_phantom(spec)
    result = measure_stenosis(volume, spec.spacing_mm)
    lumen_counts = np.count_nonzero(volume == spec.lumen_value, axis=(1, 2))

    assert abs(result.min_slice_index - spec.length // 2) <= 1
    assert lumen_counts[result.min_slice_index] == lumen_counts.min()


@pytest.mark.parametrize("min_radius", [9.0, 6.0, 3.0])
def test_VS005_accuracy_noise_free(min_radius: float) -> None:
    spec = VesselSpec(min_radius=min_radius)
    result = measure_stenosis(generate_phantom(spec), spec.spacing_mm)
    assert result.percent_stenosis == pytest.approx(spec.true_stenosis_percent, abs=3)


@pytest.mark.parametrize("min_radius", [9.0, 6.0, 3.0])
def test_VS006_accuracy_with_seeded_noise(min_radius: float) -> None:
    spec = VesselSpec(min_radius=min_radius, noise_sigma=100.0, seed=42)
    result = measure_stenosis(generate_phantom(spec), spec.spacing_mm)
    assert result.percent_stenosis == pytest.approx(spec.true_stenosis_percent, abs=3)


def test_VS007_seeded_output_is_reproducible() -> None:
    spec = VesselSpec(noise_sigma=100.0, seed=42)
    first = generate_phantom(spec)
    second = generate_phantom(spec)

    np.testing.assert_array_equal(first, second)
    assert measure_stenosis(first, spec.spacing_mm) == measure_stenosis(second, spec.spacing_mm)


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("min_radius", -1.0),
        ("min_radius", 0.0),
        ("min_radius", 12.0),
        ("min_radius", 13.0),
        ("size_xy", 23),
        ("stenosis_length", -1.0),
        ("stenosis_length", 0.0),
        ("stenosis_length", 49.0),
        ("spacing_mm", -0.5),
        ("spacing_mm", 0.0),
        ("noise_sigma", -1.0),
    ],
    ids=[
        "negative-min-radius",
        "zero-min-radius",
        "equal-radii",
        "min-radius-above-reference",
        "cross-section-too-small",
        "negative-stenosis-length",
        "zero-stenosis-length",
        "stenosis-length-too-long",
        "negative-spacing",
        "zero-spacing",
        "negative-noise",
    ],
)
def test_VS008_invalid_phantom_parameters(parameter: str, value: int | float) -> None:
    with pytest.raises(ValueError):
        VesselSpec(**{parameter: value})


def test_VS009_no_lumen_raises_value_error() -> None:
    volume = np.zeros((21, 5, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="No lumen"):
        measure_stenosis(volume, spacing_mm=0.5)


def test_VS010_spacing_scales_diameters_without_changing_percent() -> None:
    volume = generate_phantom()
    original = measure_stenosis(volume, spacing_mm=0.5)
    scaled = measure_stenosis(volume, spacing_mm=1.5)

    assert scaled.reference_diameter_mm == pytest.approx(3 * original.reference_diameter_mm)
    assert scaled.min_diameter_mm == pytest.approx(3 * original.min_diameter_mm)
    assert scaled.percent_stenosis == pytest.approx(original.percent_stenosis)
    assert scaled.min_slice_index == original.min_slice_index


@pytest.mark.parametrize(
    ("shape", "spacing_mm", "message"),
    [
        ((25, 5), 0.5, "three-dimensional"),
        ((19, 5, 5), 0.5, "at least 20 slices"),
        ((25, 5, 5), 0.0, "spacing_mm"),
        ((25, 5, 5), -0.5, "spacing_mm"),
    ],
    ids=["non-3d", "too-few-slices", "zero-spacing", "negative-spacing"],
)
def test_invalid_measurement_inputs(
    shape: tuple[int, ...], spacing_mm: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        measure_stenosis(np.zeros(shape, dtype=np.float32), spacing_mm)

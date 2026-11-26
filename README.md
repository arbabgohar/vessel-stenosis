# Vessel Stenosis Measurement

I’m a quality and software V&V engineer in medical devices, and I built this project to learn the ImFusion Python SDK and practice tracing requirements to automated tests.

I generate a synthetic 3D vessel with a known narrowing, then use NumPy to threshold each slice and estimate its equivalent circular diameter. I compare the smallest diameter with the median diameter from the first and last ten slices to calculate percent diameter stenosis.

*Screenshot placeholder for the phantom in the ImFusion viewer.*

## Run it

From the repository root in PowerShell, create a virtual environment if you don’t already have one and install the dependencies.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy pytest
.\.venv\Scripts\python.exe -m pip install "imfusion-sdk[webshop-base]==0.12.0" --extra-index-url https://pypi.imfusion.com/simple
```

For the ImFusion test and viewer, set `IMFUSION_LICENSE_KEY` in the environment where you run Python. Then run the suite or open the phantom.

```powershell
.\.venv\Scripts\python.exe -m pytest -v -rs
.\.venv\Scripts\python.exe scripts/show_phantom.py
```

## Requirements and tests

| ID | Requirement | Test |
| --- | --- | --- |
| VS-001 | Generate a straight vessel along axis 0 with known reference and minimum radii. | `test_VS001_phantom_geometry` |
| VS-002 | Calculate ground truth as `(1 - min_radius / ref_radius) * 100`. | `test_VS002_ground_truth` |
| VS-003 | Report the reference diameter in mm within 5% of the expected diameter. | `test_VS003_reference_diameter_in_mm` |
| VS-004 | Locate the narrowest slice within one slice of the center. | `test_VS004_narrowest_slice` |
| VS-005 | Measure 25%, 50%, and 75% stenosis within ±3 percentage points without noise. | `test_VS005_accuracy_noise_free` |
| VS-006 | Meet the same tolerance with seeded Gaussian noise at σ = 100. | `test_VS006_accuracy_with_seeded_noise` |
| VS-007 | Reproduce the volume and measurement for identical inputs and seed. | `test_VS007_seeded_output_is_reproducible` |
| VS-008 | Reject invalid phantom parameters with `ValueError`. | `test_VS008_invalid_phantom_parameters` |
| VS-009 | Raise `ValueError` for a volume with no lumen. | `test_VS009_no_lumen_raises_value_error` |
| VS-010 | Scale diameters with voxel spacing while keeping percent stenosis constant. | `test_VS010_spacing_scales_diameters_without_changing_percent` |
| VS-011 | Preserve array axis order and voxel spacing through a `SharedImage` round trip. | `test_VS011_shared_image_preserves_axis_order_and_spacing` |

VS-001 through VS-010 are in [`tests/test_stenosis.py`](tests/test_stenosis.py). VS-011 is in [`tests/test_imfusion_io.py`](tests/test_imfusion_io.py).

## Limits

This is a portfolio project for a straight, synthetic vessel with a circular cross-section and uniform voxel spacing. The measurement uses NumPy. I use ImFusion for image conversion and visualization. The viewer is a manual check. This is not a medical device.

## Things I ran into

- Voxelization gives the narrowing a short plateau where several slices have the same measured diameter. I pick the middle slice among the tied minima.
- The installed SDK documents image buffers in `(z, y, x, channels)` order and spacing in `(x, y, z)` order. I wrote the round-trip test with distinct dimensions and spacing values so an axis swap would be visible.
- The NumPy suite passed all 30 tests in 1.90 seconds. After adding ImFusion, the run finished with 30 passed and 1 skipped in 1.45 seconds. The SDK reported a missing license during import, so the round trip and a successful viewer launch still need verification in a licensed session.

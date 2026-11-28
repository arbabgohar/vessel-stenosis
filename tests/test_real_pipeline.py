import runpy
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from curved_stenosis import measure_curved_stenosis


def test_RD006_measures_only_the_component_at_the_path_in_mm():
    z, y, _ = np.ogrid[:41, :51, :32]
    vessel = (z - 20)**2 + (y - 17)**2 <= 6**2
    neighbor = (z - 20)**2 + (y - 35)**2 <= 8**2
    mask = np.broadcast_to(vessel | neighbor, (41, 51, 32))
    isolated = np.broadcast_to(vessel, mask.shape)
    x = np.arange(2, 30)
    path = np.column_stack((np.full_like(x, 20), np.full_like(x, 17), x))
    options = dict(threshold=0.5, section_size=101, require_center_pixel=True)

    result = measure_curved_stenosis(mask, path, 0.4, **options)
    isolated_result = measure_curved_stenosis(isolated, path, 0.4, **options)

    np.testing.assert_array_equal(result.diameter_mm, isolated_result.diameter_mm)
    assert result.reference_diameter_mm == pytest.approx(4.8, abs=0.2)

    off_center_path = path + (0, -7, 0)
    assert measure_curved_stenosis(mask, off_center_path, 0.4, threshold=0.5).reference_diameter_mm > 0
    with pytest.raises(ValueError, match="No lumen found"):
        measure_curved_stenosis(mask, off_center_path, 0.4, **options)


def test_RD007_anisotropic_nifti_case_pipeline(tmp_path, monkeypatch, capsys):
    z, y, x = np.ogrid[:25, :47, :72]
    radius_mm = 5.6 - 2.8 * np.clip(1 - (abs(x - 36) - 3) / 10, 0, 1)
    vessel = ((z - 12) * 0.8)**2 + ((y - 23) * 0.4)**2 <= radius_mm**2
    ct = np.where(vessel, 350, -500).astype(np.int16)
    labels = np.where(vessel, 2, 0).astype(np.uint8)
    affine = np.diag([0.8, 0.4, 0.8, 1])
    image_path, mask_path = tmp_path / "ct.nii.gz", tmp_path / "mask.nii.gz"
    for path, data in ((image_path, ct), (mask_path, labels)):
        nifti = nib.Nifti1Image(data.transpose(2, 1, 0), affine)
        nifti.header.set_xyzt_units("mm")
        nib.save(nifti, path)

    output_dir = tmp_path / "measurements"
    script = Path(__file__).resolve().parents[1] / "scripts" / "measure_case.py"
    monkeypatch.setattr("sys.argv", [
        str(script), "--image", str(image_path), "--mask", str(mask_path), "--labels", "2",
        "--start", "12", "23", "6", "--end", "12", "23", "66", "--output-dir", str(output_dir),
    ])
    runpy.run_path(str(script), run_name="__main__")

    output = capsys.readouterr().out
    assert "Label values found: [0, 2]" in output
    assert "Input voxel spacing (z, y, x), mm: 0.8, 0.4, 0.8" in output
    assert "Resampled shape (z, y, x): (50, 47, 144)" in output
    assert "Path length: 48.00 mm" in output
    for field in ("Minimum diameter:", "Reference diameter:", "Narrowest point:"):
        assert field in output
    narrowing = float(output.split("Diameter narrowing: ")[1].split("%")[0])
    assert narrowing == pytest.approx(50, abs=5)

    profile = np.genfromtxt(output_dir / "diameter_profile.csv", delimiter=",", names=True)
    assert profile["distance_mm"][-1] == pytest.approx(48)
    assert profile["x_voxel"][np.argmin(profile["diameter_mm"])] == pytest.approx(72, abs=10)

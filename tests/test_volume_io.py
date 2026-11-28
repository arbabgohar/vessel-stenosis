import nibabel as nib
import numpy as np
import pytest

from volume_io import load_nifti_pair, resample_isotropic


def write_pair(tmp_path, image, mask, affine):
    paths = tmp_path / "image.nii.gz", tmp_path / "mask.nii.gz"
    for path, data in zip(paths, (image, mask)):
        volume = nib.Nifti1Image(data, affine)
        volume.header.set_xyzt_units("mm")
        nib.save(volume, path)
    return paths


def test_RD001_reorders_flipped_and_permuted_axes(tmp_path):
    image = np.zeros((3, 4, 5), dtype=np.int16)
    image[1, 0, 3] = 725
    affine = np.array([[0, -0.5, 0, 4], [0, 0, 0.75, 0], [2, 0, 0, 0], [0, 0, 0, 1]])
    paths = write_pair(tmp_path, image, (image > 0).astype(np.uint8), affine)

    ct, mask, spacing, _ = load_nifti_pair(*paths, vessel_labels=[1])

    assert ct.shape == mask.shape == (3, 5, 4)
    assert spacing == pytest.approx((2, 0.75, 0.5))
    assert ct[1, 3, 3] == 725
    np.testing.assert_array_equal(np.argwhere(mask), [[1, 3, 3]])


def test_RD002_resamples_at_requested_spacing_with_binary_mask(tmp_path):
    image = np.broadcast_to(np.arange(4, dtype=np.int16) * 100, (6, 5, 4)).copy()
    mask = (image >= 100) & (image <= 200)
    paths = write_pair(tmp_path, image, mask.astype(np.uint8), np.diag([0.8, 0.625, 0.5, 1]))
    ct, foreground, spacing, _ = load_nifti_pair(*paths, vessel_labels=[1])

    resampled, binary = resample_isotropic(ct, foreground, spacing)

    assert resampled.shape == binary.shape == (5, 8, 12)
    np.testing.assert_array_less(abs(np.array(resampled.shape) * 0.4 - np.array(ct.shape) * spacing), 0.4)
    np.testing.assert_allclose(resampled[:, 0, 0], [0, 80, 160, 240, 300])
    np.testing.assert_array_equal(binary[:, 0, 0], [False, True, True, True, False])
    assert binary.dtype == np.bool_


def test_RD003_reports_values_and_selects_explicit_foreground(tmp_path):
    image = np.zeros((2, 3, 4), dtype=np.int16)
    labels = image.copy()
    labels[0, 0, :3] = [1, 2, 3]
    paths = write_pair(tmp_path, image, labels, np.eye(4))

    _, mask, _, found = load_nifti_pair(*paths, vessel_labels=[1, 3])

    np.testing.assert_array_equal(found, [0, 1, 2, 3])
    np.testing.assert_array_equal(mask[:, 0, 0], [True, False, True, False])


@pytest.mark.parametrize("selection", [[], [2], [0, 1]])
def test_RD003_rejects_invalid_label_selection(tmp_path, selection):
    image = np.zeros((2, 3, 4), dtype=np.int16)
    labels = image.copy()
    labels[0, 0, 0] = 1
    paths = write_pair(tmp_path, image, labels, np.eye(4))

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        load_nifti_pair(*paths, vessel_labels=selection)


def test_RD003_rejects_empty_mask(tmp_path):
    image = np.zeros((2, 3, 4), dtype=np.int16)
    paths = write_pair(tmp_path, image, image, np.eye(4))

    with pytest.raises(ValueError, match=r"\[0\]"):
        load_nifti_pair(*paths, vessel_labels=[1])


def test_RD003_rejects_shape_mismatch(tmp_path):
    image = np.zeros((2, 3, 4), dtype=np.int16)
    paths = write_pair(tmp_path, image, image[:, :, :2], np.eye(4))

    with pytest.raises(ValueError, match="same shape"):
        load_nifti_pair(*paths, vessel_labels=[1])

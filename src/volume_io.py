from collections.abc import Sequence
from pathlib import Path

import nibabel as nib
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import affine_transform


def load_nifti_pair(
    image_path: str | Path,
    mask_path: str | Path,
    vessel_labels: Sequence[float],
) -> tuple[NDArray[np.float32], NDArray[np.bool_], tuple[float, float, float], NDArray]:
    image = nib.load(image_path)
    mask = nib.load(mask_path)
    if any(len(volume.shape) != 3 or not all(volume.shape) for volume in (image, mask)):
        raise ValueError("CT and mask must be nonempty three-dimensional volumes.")

    # Canonical NIfTI axes run right, anterior, superior. Reverse their order for z, y, x.
    image = nib.as_closest_canonical(image)
    mask = nib.as_closest_canonical(mask)
    if image.shape != mask.shape:
        raise ValueError("CT and mask must have the same shape after reorientation.")

    unit_factors = {"meter": 1000, "mm": 1, "micron": 0.001, "unknown": 1}
    image_unit = image.header.get_xyzt_units()[0]
    mask_unit = mask.header.get_xyzt_units()[0]
    image_grid = image.affine[:3] * unit_factors[image_unit]
    mask_grid = mask.affine[:3] * unit_factors[image_unit if mask_unit == "unknown" else mask_unit]
    if not np.allclose(image_grid, mask_grid, rtol=0, atol=1e-4):
        raise ValueError("CT and mask must use the same voxel grid.")

    spacing = np.linalg.norm(image_grid[:, :3], axis=0)[::-1]
    if not np.all(np.isfinite(spacing) & (spacing > 0)):
        raise ValueError("Voxel spacing must be finite and positive.")

    labels = np.asanyarray(mask.dataobj).transpose(2, 1, 0)
    found_labels = np.unique(labels)
    selected = np.asarray(vessel_labels)
    if selected.ndim != 1 or selected.size == 0 or not np.all(np.isin(selected, found_labels)):
        raise ValueError(f"Explicitly select vessel labels from the mask values: {found_labels.tolist()}.")
    binary = np.isin(labels, selected)
    if not binary.any() or binary.all():
        raise ValueError(f"Mask selection must leave foreground and background. Found values: {found_labels.tolist()}.")

    return image.get_fdata(dtype=np.float32).transpose(2, 1, 0), binary, tuple(spacing), found_labels


def resample_isotropic(
    image: NDArray[np.number],
    mask: NDArray[np.bool_],
    spacing_mm: Sequence[float],
    target_spacing_mm: float = 0.4,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    if image.ndim != 3 or not image.size or image.shape != mask.shape:
        raise ValueError("CT and mask must have the same nonempty three-dimensional shape.")
    spacing = np.asarray(spacing_mm, dtype=float)
    all_spacings = np.append(spacing, target_spacing_mm)
    if spacing.shape != (3,) or not np.all(np.isfinite(all_spacings) & (all_spacings > 0)):
        raise ValueError("Voxel spacing must be finite and positive, with three input spacings.")

    scale = spacing / target_spacing_mm
    shape = tuple(np.maximum(1, np.rint(np.asarray(image.shape) * scale).astype(int)))
    # Keep the first voxel center fixed and use the exact requested sampling interval.
    transform = np.diag(1 / scale)
    resampled_image = affine_transform(
        image, transform, output_shape=shape, output=np.float32, order=1, mode="nearest"
    )
    resampled_mask = affine_transform(
        mask, transform, output_shape=shape, output=bool, order=0, mode="nearest"
    )
    return resampled_image, resampled_mask

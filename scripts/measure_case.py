import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from centerline import resample_centerline
from curved_stenosis import measure_curved_stenosis
from vessel_path import extract_centerline
from volume_io import load_nifti_pair, resample_isotropic


SECTION_SIZE = 101


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Measure the geometry of a vessel segmentation.")
    parser.add_argument("--image", type=Path, required=True, help="CT NIfTI file")
    parser.add_argument("--mask", type=Path, required=True, help="Label NIfTI file")
    parser.add_argument("--labels", type=float, nargs="+", required=True, help="Values to treat as vessel")
    parser.add_argument("--start", type=int, nargs=3, required=True, metavar=("Z", "Y", "X"),
                        help="Start voxel in the canonical volume before resampling")
    parser.add_argument("--end", type=int, nargs=3, required=True, metavar=("Z", "Y", "X"),
                        help="End voxel in the canonical volume before resampling")
    parser.add_argument("--spacing", type=float, default=0.4, help="Isotropic voxel spacing in mm (default 0.4)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for the diameter profile CSV")
    args = parser.parse_args(argv)

    try:
        ct, mask, spacing, labels = load_nifti_pair(args.image, args.mask, args.labels)
        endpoints = np.array([args.start, args.end])
        for name, point in zip(("Start", "End"), endpoints):
            if (point < 0).any() or (point >= mask.shape).any() or not mask[tuple(point)]:
                raise ValueError(f"{name} is outside the vessel mask before resampling.")

        isotropic_ct, isotropic_mask = resample_isotropic(ct, mask, spacing, args.spacing)
        del ct, mask, isotropic_ct
        # Resampling preserves the first voxel center, so only per-axis scale changes the indices.
        start, end = np.rint(endpoints * np.asarray(spacing) / args.spacing).astype(np.intp)
        path = extract_centerline(isotropic_mask, start, end, args.spacing)
        result = measure_curved_stenosis(
            isotropic_mask, path, args.spacing, threshold=0.5,
            section_size=SECTION_SIZE, require_center_pixel=True,
        )
        points, _ = resample_centerline(path, args.spacing, step_mm=args.spacing)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            args.output_dir / "diameter_profile.csv",
            np.column_stack((result.distance_mm, result.diameter_mm, points)),
            delimiter=",", header="distance_mm,diameter_mm,z_voxel,y_voxel,x_voxel", comments="",
        )
    except (OSError, ValueError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")

    minimum = np.searchsorted(result.distance_mm, result.min_distance_mm)
    location = ", ".join(f"{coordinate:.2f}" for coordinate in points[minimum])
    print(f"Label values found: {labels.tolist()}")
    print(f"Input voxel spacing (z, y, x), mm: {', '.join(f'{value:.6g}' for value in spacing)}")
    print(f"Resampled voxel spacing: {args.spacing:g} mm isotropic")
    print(f"Resampled shape (z, y, x): {isotropic_mask.shape}")
    print(f"Path length: {result.distance_mm[-1]:.2f} mm")
    print(f"Minimum diameter: {result.min_diameter_mm:.2f} mm")
    print(f"Reference diameter: {result.reference_diameter_mm:.2f} mm")
    print(f"Narrowest point: {result.min_distance_mm:.2f} mm from start, resampled (z, y, x) = ({location})")
    print(f"Diameter narrowing: {result.percent_stenosis:.2f}%")


if __name__ == "__main__":
    main()

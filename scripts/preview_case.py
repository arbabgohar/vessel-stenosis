import sys

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Line3DCollection

sys.dont_write_bytecode = True

from measure_case import ROOT, SECTION_SIZE, case_parser, load_case_path, measure_mask_path
from centerline import resample_centerline, sample_cross_sections


def make_figure(mask, path, spacing_mm):
    result = measure_mask_path(mask, path, spacing_mm)
    pixel_spacing_mm = spacing_mm / 2
    sections, distances = sample_cross_sections(
        mask, path, spacing_mm, step_mm=spacing_mm,
        section_size=SECTION_SIZE, pixel_spacing_mm=pixel_spacing_mm,
    )
    points, _ = resample_centerline(path, spacing_mm, step_mm=spacing_mm)

    figure = Figure(figsize=(15, 4.8), layout="constrained")
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(1, 3, width_ratios=(1, 1, 1.15))
    straightened = figure.add_subplot(grid[0])
    profile = figure.add_subplot(grid[1], sharex=straightened)
    arc = figure.add_subplot(grid[2], projection="3d")

    half_step = (distances[1] - distances[0]) / 2
    half_width = SECTION_SIZE * pixel_spacing_mm / 2
    minimum_color = "#ae2435"
    straightened.imshow(
        (sections[:, SECTION_SIZE // 2, :] > 0.5).T, cmap="gray",
        extent=(-half_step, distances[-1] + half_step, -half_width, half_width),
        origin="lower", aspect="auto", interpolation="nearest", vmin=0, vmax=1,
    )
    straightened.axvline(result.min_distance_mm, color="#ff666e", linestyle="--")
    straightened.set(title="Straightened vessel mask", xlabel="Distance along path (mm)", ylabel="Offset (mm)")

    profile.plot(result.distance_mm, result.diameter_mm, color="#245c91", linewidth=1.8)
    profile.axhline(result.reference_diameter_mm, color="#626971", linestyle="--",
                    label=f"Reference {result.reference_diameter_mm:.2f} mm")
    profile.scatter(result.min_distance_mm, result.min_diameter_mm, color=minimum_color, zorder=3,
                    label=f"Minimum {result.min_diameter_mm:.2f} mm  |  {result.percent_stenosis:.1f}% narrowing")
    profile.set(title="Diameter along path", xlabel="Distance along path (mm)", ylabel="Diameter (mm)",
                xlim=(0, distances[-1]), ylim=(0, result.diameter_mm.max() * 1.2))
    profile.grid(alpha=0.22)
    profile.legend(loc="best", framealpha=0.9, fontsize=9)

    xyz = points[:, ::-1] * spacing_mm
    segments = np.stack((xyz[:-1], xyz[1:]), axis=1)
    low, high = result.min_diameter_mm, result.diameter_mm.max()
    if high == low:
        low, high = max(0, low - 0.01), high + 0.01
    line = Line3DCollection(segments, cmap="RdBu", norm=Normalize(low, high), linewidths=4, capstyle="round")
    line.set_array((result.diameter_mm[:-1] + result.diameter_mm[1:]) / 2)
    arc.add_collection3d(line)
    arc.set_title("3D centerline")
    for axis, name, bins in zip((arc.xaxis, arc.yaxis, arc.zaxis), "XYZ", (4, 2, 3)):
        axis.set_label_text(f"{name} (mm)")
        axis.labelpad = 12
        axis.set_major_locator(MaxNLocator(nbins=bins))
        axis.set_tick_params(labelsize=8)
    midpoints = (xyz.max(axis=0) + xyz.min(axis=0)) / 2
    spans = np.ptp(xyz, axis=0)
    half_spans = np.maximum(spans * 0.55, spans.max() * 0.15)
    for center, half_span, set_limit in zip(midpoints, half_spans, (arc.set_xlim, arc.set_ylim, arc.set_zlim)):
        set_limit(center - half_span, center + half_span)
    arc.set_box_aspect(half_spans, zoom=0.84)
    figure.colorbar(line, ax=arc, orientation="horizontal", shrink=0.8, pad=0.04, label="Diameter (mm)")
    return figure


def main(argv=None) -> None:
    parser = case_parser("Plot the geometry of a measured vessel segmentation.", "Folder outside the repository for the PNG")
    args = parser.parse_args(argv)
    try:
        output = (args.output_dir / "vessel_preview.png").resolve()
        if output.is_relative_to(ROOT):
            raise ValueError("Choose an output folder outside the repository.")
        mask, path, _, _ = load_case_path(args)
        figure = make_figure(mask, path, args.spacing)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=160)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()

import sys
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from centerline import resample_centerline, sample_cross_sections
from curved_stenosis import measure_curved_stenosis
from curved_vessel_phantom import CurvedVesselSpec, generate_curved_phantom, true_centerline


def main() -> None:
    spec = CurvedVesselSpec(noise_sigma=100.0, seed=42)
    volume = generate_curved_phantom(spec)
    centerline = true_centerline(spec)
    section_size = 41
    pixel_spacing_mm = 0.25
    sampling = dict(
        step_mm=spec.spacing_mm,
        section_size=section_size,
        pixel_spacing_mm=pixel_spacing_mm,
    )
    sections, distance_mm = sample_cross_sections(volume, centerline, spec.spacing_mm, **sampling)
    result = measure_curved_stenosis(volume, centerline, spec.spacing_mm, **sampling)
    points, _ = resample_centerline(centerline, spec.spacing_mm, step_mm=spec.spacing_mm)

    figure = Figure(figsize=(14, 4.7), layout="constrained")
    FigureCanvasAgg(figure)
    figure.suptitle("Curved vessel stenosis · seeded synthetic noise (σ = 100)", fontsize=15)
    gridspec = figure.add_gridspec(1, 3, width_ratios=(1, 1, 1.15))
    straightened = figure.add_subplot(gridspec[0])
    profile = figure.add_subplot(gridspec[1], sharex=straightened)
    arc = figure.add_subplot(gridspec[2], projection="3d")

    minimum_color = "#b52b32"
    half_step = (distance_mm[1] - distance_mm[0]) / 2
    half_width = section_size * pixel_spacing_mm / 2
    straightened.imshow(
        sections[:, section_size // 2, :].T,
        extent=(-half_step, distance_mm[-1] + half_step, -half_width, half_width),
        origin="lower",
        interpolation="nearest",
        aspect="auto",
        cmap="gray",
        vmin=spec.background_value,
        vmax=spec.lumen_value,
    )
    straightened.axvline(result.min_distance_mm, color="#ff6b6b", linewidth=1.5, linestyle="--")
    straightened.set(
        title="Straightened vessel", xlabel="Distance along vessel (mm)", ylabel="Offset (mm)",
        xlim=(0, distance_mm[-1]),
    )

    profile.plot(result.distance_mm, result.diameter_mm, color="#245c91", linewidth=2)
    profile.axhline(
        result.reference_diameter_mm, color="#59636e", linestyle="--",
        label=f"Reference {result.reference_diameter_mm:.2f} mm",
    )
    profile.scatter(result.min_distance_mm, result.min_diameter_mm, color=minimum_color, zorder=3)
    profile.annotate(
        f"Minimum {result.min_diameter_mm:.2f} mm\n"
        f"Measured {result.percent_stenosis:.1f}% · true {spec.true_stenosis_percent:.1f}%",
        xy=(result.min_distance_mm, result.min_diameter_mm),
        xytext=(0, -17),
        textcoords="offset points",
        ha="center",
        va="top",
        color=minimum_color,
    )
    profile.set(
        title="Diameter profile", xlabel="Distance along vessel (mm)", ylabel="Diameter (mm)",
        ylim=(0, 1.2 * result.diameter_mm.max()),
    )
    profile.grid(alpha=0.22)
    profile.legend(loc="upper right", frameon=False)

    zyx_mm = points * spec.spacing_mm
    segments = np.stack((zyx_mm[:-1], zyx_mm[1:]), axis=1)
    colors = Line3DCollection(
        segments,
        cmap="coolwarm_r",
        norm=Normalize(result.min_diameter_mm, result.diameter_mm.max()),
        linewidths=5,
    )
    colors.set_array((result.diameter_mm[:-1] + result.diameter_mm[1:]) / 2)
    arc.add_collection3d(colors)
    z, y, x = zyx_mm.T
    arc.auto_scale_xyz(z, [y.min() - 4, y.max() + 4], x)
    arc.set_box_aspect((np.ptp(z), 8, np.ptp(x)), zoom=0.85)
    arc.view_init(elev=25, azim=-68)
    arc.set(title="3D centerline", xticks=[20, 30, 50, 60], yticks=[40], zticks=[38, 42])
    arc.set_xlabel("Z (mm)", labelpad=10)
    arc.set_ylabel("Y (mm)", labelpad=8)
    arc.set_zlabel("X (mm)", labelpad=8)
    figure.colorbar(
        colors, ax=arc, orientation="horizontal", shrink=0.75, pad=0.04,
        label="Diameter (mm)",
    )

    output = ROOT / "docs" / "curved_preview.png"
    figure.savefig(output, dpi=160)
    print(
        f"Measured stenosis {result.percent_stenosis:.2f}% "
        f"(true {spec.true_stenosis_percent:.2f}%). "
        f"Minimum diameter {result.min_diameter_mm:.2f} mm at {result.min_distance_mm:.2f} mm."
    )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()

import argparse
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Slider

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

from volume_io import load_nifti_pair


def populate_slice_figure(figure, ct, mask, spacing, on_selection=None):
    position = [length // 2 for length in ct.shape]
    low, high = np.percentile(ct[::4, ::4, ::4], (2, 98))
    axes = figure.subplots(1, 3)
    figure.subplots_adjust(left=0.07, right=0.97, bottom=0.3, top=0.81, wspace=0.32)
    summary = figure.suptitle("")
    figure.text(0.5, 0.92, "Click a slice to select a voxel. Use the sliders to move through the volume.", ha="center")
    views = (("Axial", 0, 1, 2), ("Coronal", 1, 0, 2), ("Sagittal", 2, 0, 1))
    axis_names = ("Z", "Y", "X")
    overlay_colors = ListedColormap([(0, 0, 0, 0), (1, 0, 0, 0.45)])
    artists = []
    sliders = []

    for ax, (title, fixed, row, col) in zip(axes, views):
        plane = np.take(ct, position[fixed], axis=fixed)
        image = ax.imshow(plane, cmap="gray", origin="lower", interpolation="nearest",
                          vmin=low, vmax=high, aspect=spacing[row] / spacing[col])
        overlay = ax.imshow(np.take(mask, position[fixed], axis=fixed), cmap=overlay_colors,
                            origin="lower", interpolation="nearest", vmin=0, vmax=1,
                            aspect=spacing[row] / spacing[col])
        vertical = ax.axvline(position[col], color="cyan", linewidth=0.7)
        horizontal = ax.axhline(position[row], color="cyan", linewidth=0.7)
        ax.set(title=title, xlabel=f"{axis_names[col]} voxel", ylabel=f"{axis_names[row]} voxel")
        artists.append((image, overlay, vertical, horizontal))

        slider_axis = figure.add_axes((0.17, 0.19 - 0.06 * fixed, 0.66, 0.03))
        slider = Slider(slider_axis, axis_names[fixed], 0, max(1, ct.shape[fixed] - 1),
                        valinit=position[fixed], valstep=1, valfmt="%d")
        if ct.shape[fixed] == 1:
            slider.set_active(False)
        sliders.append(slider)

    def redraw() -> None:
        for (_, fixed, row, col), (image, overlay, vertical, horizontal) in zip(views, artists):
            image.set_data(np.take(ct, position[fixed], axis=fixed))
            overlay.set_data(np.take(mask, position[fixed], axis=fixed))
            vertical.set_xdata([position[col], position[col]])
            horizontal.set_ydata([position[row], position[row]])
        inside = "inside vessel" if mask[tuple(position)] else "outside vessel"
        summary.set_text(f"Voxel (z, y, x) = {tuple(position)}  |  {inside}")
        if on_selection is not None:
            on_selection(tuple(position), bool(mask[tuple(position)]))
        figure.canvas.draw_idle()

    def move(index, value) -> None:
        position[index] = int(value)
        redraw()

    def pick(event) -> None:
        if event.button != 1 or event.inaxes not in axes or event.xdata is None or event.ydata is None:
            return
        toolbar = figure.canvas.toolbar
        if toolbar is not None and toolbar.mode:
            return
        _, _, row, col = views[list(axes).index(event.inaxes)]
        for index, coordinate in ((row, event.ydata), (col, event.xdata)):
            voxel = int(np.clip(round(coordinate), 0, ct.shape[index] - 1))
            sliders[index].set_val(voxel)
        if on_selection is None:
            print("Selected voxel (z y x):", *position, flush=True)

    for index, slider in enumerate(sliders):
        slider.on_changed(lambda value, axis=index: move(axis, value))
    figure.canvas.mpl_connect("button_press_event", pick)
    redraw()
    return sliders


def show_slices(ct, mask, spacing) -> None:
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(13, 6))
    sliders = populate_slice_figure(figure, ct, mask, spacing)
    plt.show()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Read endpoint voxels from orthogonal CT slices.")
    parser.add_argument("--image", type=Path, required=True, help="CT NIfTI file")
    parser.add_argument("--mask", type=Path, required=True, help="Label NIfTI file")
    parser.add_argument("--labels", type=float, nargs="+", required=True, help="Values to display as vessel")
    args = parser.parse_args(argv)

    try:
        ct, mask, spacing, labels = load_nifti_pair(args.image, args.mask, args.labels)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
    print(f"Label values found: {labels.tolist()}")
    show_slices(ct, mask, spacing)


if __name__ == "__main__":
    main()

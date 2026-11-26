from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class CurvedVesselSpec:
    size: int = 160
    ref_radius: float = 8.0
    min_radius: float = 4.0
    bend_radius: float = 60.0
    sweep_deg: float = 90.0
    stenosis_length: float = 24.0
    lumen_value: float = 1000.0
    background_value: float = 0.0
    spacing_mm: float = 0.5
    noise_sigma: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not np.isfinite(self.ref_radius) or not 0 < self.min_radius < self.ref_radius:
            raise ValueError("Radii must satisfy 0 < min_radius < ref_radius.")
        if not np.isfinite(self.bend_radius) or self.bend_radius <= 0:
            raise ValueError("bend_radius must be finite and positive.")
        if not 0 < self.sweep_deg < 360:
            raise ValueError("sweep_deg must be between 0 and 360 degrees.")
        half_sweep = np.deg2rad(self.sweep_deg / 2)
        if not 0 < self.stenosis_length <= self.bend_radius * half_sweep:
            raise ValueError("stenosis_length must fit on either side of the arc midpoint.")
        if not np.isfinite(self.spacing_mm) or self.spacing_mm <= 0:
            raise ValueError("spacing_mm must be finite and positive.")
        if not np.isfinite(self.noise_sigma) or self.noise_sigma < 0:
            raise ValueError("noise_sigma must be finite and nonnegative.")
        if not isinstance(self.size, (int, np.integer)) or self.size <= 0:
            raise ValueError("size must be a positive integer.")

        z_extent = 2 * self.bend_radius * np.sin(min(half_sweep, np.pi / 2))
        x_extent = self.bend_radius * (1 - np.cos(half_sweep))
        if max(z_extent, x_extent) + 2 * (self.ref_radius + 2) > self.size - 1:
            raise ValueError("size must accommodate the arc with a ref_radius + 2 voxel margin.")

    @property
    def true_stenosis_percent(self) -> float:
        return (1 - self.min_radius / self.ref_radius) * 100


def true_centerline(spec: CurvedVesselSpec = CurvedVesselSpec()) -> NDArray[np.float64]:
    half_sweep = np.deg2rad(spec.sweep_deg / 2)
    half_steps = int(np.ceil(spec.bend_radius * half_sweep / 0.1))
    angles = np.linspace(-half_sweep, half_sweep, 2 * half_steps + 1)
    middle = (spec.size - 1) / 2
    x_midrange = (1 + np.cos(half_sweep)) / 2
    return np.column_stack(
        (
            middle + spec.bend_radius * np.sin(angles),
            np.full(angles.shape, middle),
            middle + spec.bend_radius * (np.cos(angles) - x_midrange),
        )
    )


def generate_curved_phantom(
    spec: CurvedVesselSpec = CurvedVesselSpec(),
) -> NDArray[np.float32]:
    centerline = true_centerline(spec)
    half_length = spec.bend_radius * np.deg2rad(spec.sweep_deg / 2)
    arc_distance = np.abs(np.linspace(-half_length, half_length, len(centerline)))
    taper_position = np.minimum(arc_distance / spec.stenosis_length, 1)
    radii = spec.min_radius + (spec.ref_radius - spec.min_radius) * (
        1 - np.cos(np.pi * taper_position)
    ) / 2

    volume = np.full((spec.size,) * 3, spec.background_value, dtype=np.float32)
    tree = cKDTree(centerline)
    lower = np.floor(centerline.min(axis=0) - spec.ref_radius).astype(int)
    upper = np.ceil(centerline.max(axis=0) + spec.ref_radius).astype(int)
    y, x = np.mgrid[lower[1]:upper[1] + 1, lower[2]:upper[2] + 1]
    points = np.column_stack((np.empty(y.size), y.ravel(), x.ravel()))
    for z in range(lower[0], upper[0] + 1):
        points[:, 0] = z
        distances, nearest = tree.query(points)
        lumen = distances <= radii[nearest]
        volume[z, y.ravel()[lumen], x.ravel()[lumen]] = spec.lumen_value

    if spec.noise_sigma > 0:
        volume += np.random.default_rng(spec.seed).normal(0, spec.noise_sigma, volume.shape)
    return volume

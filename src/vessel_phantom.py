from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class VesselSpec:
    size_xy: int = 65
    length: int = 97
    ref_radius: float = 12.0
    min_radius: float = 6.0
    stenosis_length: float = 30.0
    lumen_value: float = 1000.0
    background_value: float = 0.0
    spacing_mm: float = 0.5
    noise_sigma: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0 < self.min_radius < self.ref_radius:
            raise ValueError("Radii must satisfy 0 < min_radius < ref_radius.")
        if not self.ref_radius <= (self.size_xy - 1) / 2:
            raise ValueError("size_xy must accommodate the reference radius.")
        if self.length % 2 != 1 or not 0 < self.stenosis_length <= (self.length - 1) / 2:
            raise ValueError("Use an odd length and a taper that fits on either side of center.")
        if not np.isfinite(self.spacing_mm) or self.spacing_mm <= 0:
            raise ValueError("spacing_mm must be finite and positive.")
        if not np.isfinite(self.noise_sigma) or self.noise_sigma < 0:
            raise ValueError("noise_sigma must be finite and nonnegative.")

    @property
    def true_stenosis_percent(self) -> float:
        return (1 - self.min_radius / self.ref_radius) * 100


def generate_phantom(spec: VesselSpec = VesselSpec()) -> NDArray[np.float32]:
    slice_distance = np.abs(np.arange(spec.length) - (spec.length - 1) / 2)
    taper_position = np.minimum(slice_distance / spec.stenosis_length, 1)
    radii = spec.min_radius + (spec.ref_radius - spec.min_radius) * (
        1 - np.cos(np.pi * taper_position)
    ) / 2

    offset = np.arange(spec.size_xy) - (spec.size_xy - 1) / 2
    distance_squared = offset[:, None] ** 2 + offset[None, :] ** 2
    lumen = distance_squared <= radii[:, None, None] ** 2
    volume = np.where(lumen, spec.lumen_value, spec.background_value).astype(np.float32)
    if spec.noise_sigma > 0:
        volume += np.random.default_rng(spec.seed).normal(0, spec.noise_sigma, volume.shape)
    return volume

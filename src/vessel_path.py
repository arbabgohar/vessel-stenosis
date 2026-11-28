from collections.abc import Sequence
from heapq import heappop, heappush
from itertools import product
from math import dist, isfinite, sqrt

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt


_NEIGHBORS = tuple(
    (dz, dy, dx, sqrt(dz**2 + dy**2 + dx**2))
    for dz, dy, dx in product((-1, 0, 1), repeat=3)
    if (dz, dy, dx) != (0, 0, 0)
)


def extract_centerline(
    mask: NDArray[np.bool_],
    start: Sequence[int],
    end: Sequence[int],
    spacing_mm: float,
) -> NDArray[np.intp]:
    if not isfinite(spacing_mm) or spacing_mm <= 0:
        raise ValueError("Voxel spacing must be finite and positive.")
    if mask.ndim != 3 or mask.dtype != np.bool_:
        raise ValueError("Mask must be a three-dimensional boolean array.")

    start_point = _mask_point(mask, start, "Start")
    end_point = _mask_point(mask, end, "End")
    if start_point == end_point:
        return np.array([start_point], dtype=np.intp)

    bounds = [np.flatnonzero(mask.any(axis=axes)) for axes in ((1, 2), (0, 2), (0, 1))]
    offset = tuple(int(indices[0]) for indices in bounds)
    cropped = mask[tuple(slice(low, indices[-1] + 1) for low, indices in zip(offset, bounds))]
    start_point = tuple(index - low for index, low in zip(start_point, offset))
    end_point = tuple(index - low for index, low in zip(end_point, offset))

    # Padding supplies background at the crop boundary, including when the mask fills the crop.
    clearance = distance_transform_edt(np.pad(cropped, 1))[1:-1, 1:-1, 1:-1]
    # Normalizing the inverse-square cost gives every edge a cost at least its voxel length.
    cost = np.zeros_like(clearance)
    cost[cropped] = (clearance.max() / clearance[cropped]) ** 2
    depth, height, width = cropped.shape
    queue = [(dist(start_point, end_point), 0.0, start_point)]
    best = {start_point: 0.0}
    parent = {}

    while queue:
        _, current_cost, point = heappop(queue)
        if current_cost != best[point]:
            continue
        if point == end_point:
            path = [point]
            while point != start_point:
                point = parent[point]
                path.append(point)
            return np.array(path[::-1], dtype=np.intp) + offset

        z, y, x = point
        voxel_cost = cost[point]
        for dz, dy, dx, length in _NEIGHBORS:
            nz, ny, nx = neighbor = (z + dz, y + dy, x + dx)
            if not (0 <= nz < depth and 0 <= ny < height and 0 <= nx < width and cropped[neighbor]):
                continue
            candidate = current_cost + length * (voxel_cost + cost[neighbor]) / 2
            if candidate < best.get(neighbor, float("inf")):
                best[neighbor] = candidate
                parent[neighbor] = point
                heappush(queue, (candidate + dist(neighbor, end_point), candidate, neighbor))

    raise ValueError("No connected vessel path between start and end.")


def _mask_point(mask: NDArray[np.bool_], point: Sequence[int], name: str) -> tuple[int, int, int]:
    coordinates = np.asarray(point)
    if coordinates.shape != (3,) or not np.issubdtype(coordinates.dtype, np.integer):
        raise ValueError(f"{name} must contain three integer voxel coordinates in (z, y, x) order.")
    index = tuple(map(int, coordinates))
    if any(value < 0 or value >= size for value, size in zip(index, mask.shape)) or not mask[index]:
        raise ValueError(f"{name} is outside the vessel mask.")
    return index

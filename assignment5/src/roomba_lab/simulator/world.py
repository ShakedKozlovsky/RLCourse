"""Static apartment world — polygon boundary + occupancy grid for coverage tracking.

The polygon (shapely) is the **free-space** region: inside = drivable, outside = wall.
The occupancy grid is a rasterised version used to mark which cells have been
cleaned and to compute the coverage fraction efficiently."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from shapely import prepare
from shapely.geometry import Polygon

UNVISITED = 0
VISITED = 1
OBSTACLE = 255


@dataclass
class World:
    polygon: Polygon
    pixels_per_metre: float
    grid: np.ndarray = field(init=False)
    bbox_min: tuple[float, float] = field(init=False)
    bbox_max: tuple[float, float] = field(init=False)

    def __post_init__(self) -> None:
        x_min, y_min, x_max, y_max = self.polygon.bounds
        self.bbox_min = (float(x_min), float(y_min))
        self.bbox_max = (float(x_max), float(y_max))
        w_pix = max(1, int((x_max - x_min) * self.pixels_per_metre))
        h_pix = max(1, int((y_max - y_min) * self.pixels_per_metre))
        self.grid = np.full((h_pix, w_pix), OBSTACLE, dtype=np.uint8)
        self._rasterise_free_space()
        prepare(self.polygon)

    def _rasterise_free_space(self) -> None:
        h, w = self.grid.shape
        ys = np.linspace(self.bbox_min[1], self.bbox_max[1], h, endpoint=False) + 0.5 / self.pixels_per_metre
        xs = np.linspace(self.bbox_min[0], self.bbox_max[0], w, endpoint=False) + 0.5 / self.pixels_per_metre
        from shapely.geometry import Point  # local import: shapely Point is heavy
        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                if self.polygon.contains(Point(x, y)):
                    self.grid[i, j] = UNVISITED

    def cell_index(self, x: float, y: float) -> tuple[int, int]:
        """Cell index."""
        j = int((x - self.bbox_min[0]) * self.pixels_per_metre)
        i = int((y - self.bbox_min[1]) * self.pixels_per_metre)
        h, w = self.grid.shape
        return max(0, min(h - 1, i)), max(0, min(w - 1, j))

    def free_cell_count(self) -> int:
        """Free cell count."""
        return int(np.sum((self.grid == UNVISITED) | (self.grid == VISITED)))

    def visited_cell_count(self) -> int:
        """Visited cell count."""
        return int(np.sum(self.grid == VISITED))

    def coverage_fraction(self) -> float:
        """Coverage fraction."""
        free = self.free_cell_count()
        return 0.0 if free == 0 else self.visited_cell_count() / free

    def reset_visits(self) -> None:
        """Reset visits."""
        self.grid[self.grid == VISITED] = UNVISITED

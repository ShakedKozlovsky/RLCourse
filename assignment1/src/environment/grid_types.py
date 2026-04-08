"""Grid environment type definitions."""

from enum import Enum
from dataclasses import dataclass


class CellType(Enum):
    """Grid cell types."""
    EMPTY = 0
    BUILDING = 1
    TRAP = 2
    GOAL = 3
    WIND_ZONE = 4


@dataclass
class Wind:
    """Wind vector at a grid position."""
    dx: int
    dy: int

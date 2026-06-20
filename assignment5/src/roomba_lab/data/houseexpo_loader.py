"""HouseExpo dataset loader. Parses the upstream JSON format into geometry the
simulator can consume.

A HouseExpo JSON file has the shape::

    {"verts": [[x, y], ...], "id": "<sha>", "bbox": {"min": [x,y], "max": [x,y]},
     "room_category": {"Kitchen": [...], ...}, "room_num": 6}

`verts` is the apartment's **outer boundary polygon** (in metres). The robot
lives inside that polygon; everything outside is wall."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HouseMap:
    map_id: str
    verts: list[tuple[float, float]]
    bbox_min: tuple[float, float]
    bbox_max: tuple[float, float]
    room_num: int

    @property
    def width_m(self) -> float:
        return self.bbox_max[0] - self.bbox_min[0]

    @property
    def height_m(self) -> float:
        return self.bbox_max[1] - self.bbox_min[1]


class HouseExpoLoader:
    """Reads HouseExpo JSON apartments from a local directory."""

    def __init__(self, sample_dir: Path) -> None:
        self._sample_dir = Path(sample_dir)
        if not self._sample_dir.exists():
            raise FileNotFoundError(f"HouseExpo sample dir not found: {sample_dir}")
        self._cache: dict[str, HouseMap] = {}

    def map_ids(self) -> list[str]:
        return sorted(p.stem for p in self._sample_dir.glob("*.json"))

    def load(self, map_id: str) -> HouseMap:
        if map_id in self._cache:
            return self._cache[map_id]
        path = self._sample_dir / f"{map_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Map {map_id!r} not in {self._sample_dir}")
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        verts = [(float(x), float(y)) for x, y in raw["verts"]]
        bmin = (float(raw["bbox"]["min"][0]), float(raw["bbox"]["min"][1]))
        bmax = (float(raw["bbox"]["max"][0]), float(raw["bbox"]["max"][1]))
        house = HouseMap(
            map_id=map_id,
            verts=verts,
            bbox_min=bmin,
            bbox_max=bmax,
            room_num=int(raw.get("room_num", 1)),
        )
        self._cache[map_id] = house
        return house

    @staticmethod
    def content_hash(path: Path) -> str:
        """Cache key by file SHA-256 (ADR-009 — content-addressed)."""
        h = hashlib.sha256()
        h.update(Path(path).read_bytes())
        return h.hexdigest()

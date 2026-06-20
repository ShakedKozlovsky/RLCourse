"""Layer 1 — HouseExpo loader tests against the committed 10-map sample."""

from __future__ import annotations

import json

import pytest

from roomba_lab.data.houseexpo_loader import HouseExpoLoader
from roomba_lab.shared.config import PROJECT_ROOT

SAMPLE_DIR = PROJECT_ROOT / "data" / "raw" / "sample_maps"


@pytest.fixture
def loader() -> HouseExpoLoader:
    return HouseExpoLoader(SAMPLE_DIR)


def test_sample_dir_populated() -> None:
    assert SAMPLE_DIR.exists()
    assert len(list(SAMPLE_DIR.glob("*.json"))) == 10


def test_loader_lists_ten_maps(loader: HouseExpoLoader) -> None:
    ids = loader.map_ids()
    assert len(ids) == 10
    for mid in ids:
        assert len(mid) == 32  # SHA-128 hex


def test_loader_parses_geometry(loader: HouseExpoLoader) -> None:
    house = loader.load(loader.map_ids()[0])
    assert len(house.verts) >= 4
    assert house.width_m > 0
    assert house.height_m > 0
    for x, y in house.verts:
        assert house.bbox_min[0] - 1e-3 <= x <= house.bbox_max[0] + 1e-3
        assert house.bbox_min[1] - 1e-3 <= y <= house.bbox_max[1] + 1e-3


def test_loader_caches_repeated_loads(loader: HouseExpoLoader) -> None:
    mid = loader.map_ids()[0]
    h1 = loader.load(mid)
    h2 = loader.load(mid)
    assert h1 is h2


def test_loader_missing_id_raises(loader: HouseExpoLoader) -> None:
    with pytest.raises(FileNotFoundError):
        loader.load("does_not_exist_at_all")


def test_content_hash_changes_on_edit(tmp_path) -> None:
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"verts": [[0, 0], [1, 0], [1, 1], [0, 1]],
                              "bbox": {"min": [0, 0], "max": [1, 1]},
                              "room_num": 1}))
    h1 = HouseExpoLoader.content_hash(p)
    p.write_text(p.read_text() + " ")
    h2 = HouseExpoLoader.content_hash(p)
    assert h1 != h2

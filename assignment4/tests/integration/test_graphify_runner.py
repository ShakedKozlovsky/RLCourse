"""Graphify runner — end-to-end on the real project source tree."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from proximal_lab.shared.config import ConfigManager
from proximal_lab.tools.graphify.runner import run_graphify


@pytest.fixture
def sdk_config(tmp_path: Path) -> Path:
    cfg = {
        "version": "1.00",
        "seed": 0,
        "env": {"id": "HalfCheetah-v5", "gamma": 0.99, "n_parallel_envs": 2},
        "gae": {"lambda": 0.95},
        "graphify": {"src_dir": "src/proximal_lab", "output_dir": "docs/wiki"},
        "paths": {"results_dir": str(tmp_path / "results"),
                   "assets_dir": str(tmp_path / "assets"),
                   "checkpoints_dir": str(tmp_path / "saved_models"),
                   "wiki_dir": str(tmp_path / "wiki")},
    }
    path = tmp_path / "setup.json"
    path.write_text(json.dumps(cfg))
    return path


def test_runner_emits_wiki_on_synthetic_src(sdk_config: Path, tmp_path: Path) -> None:
    src = tmp_path / "fake_src"
    src.mkdir()
    (src / "module_a.py").write_text(
        '"""Module A docstring."""\n'
        "class Foo:\n"
        '    """The Foo class."""\n'
        "    pass\n"
    )
    out = tmp_path / "wiki_out"
    cfg = ConfigManager(setup_path=sdk_config)
    run_graphify(cfg, src_dir=src, out_dir=out)
    assert (out / "graph.json").exists()
    assert (out / "_index.md").exists()
    # at least one module markdown
    md_files = list(out.glob("*.md"))
    assert any(m.name != "_index.md" for m in md_files)


def test_runner_emits_wiki_on_real_src(sdk_config: Path, tmp_path: Path) -> None:
    real_src = Path(__file__).resolve().parents[2] / "src" / "proximal_lab"
    out = tmp_path / "real_wiki"
    cfg = ConfigManager(setup_path=sdk_config)
    run_graphify(cfg, src_dir=real_src, out_dir=out)
    blob = json.loads((out / "graph.json").read_text())
    # At least 10 modules in the real project
    modules = [n for n in blob["nodes"] if n["type"] == "module"]
    assert len(modules) >= 10
    # graph.json edges include imports
    assert any(e["type"] == "imports" for e in blob["edges"])

"""Graphify CLI entry — walk src/roomba_lab and emit docs/wiki/."""

from __future__ import annotations

from pathlib import Path

from roomba_lab.shared.config import ConfigManager
from roomba_lab.shared.logger import get_logger
from roomba_lab.tools.graphify.emitter import emit
from roomba_lab.tools.graphify.walker import walk_source_tree

_logger = get_logger(__name__)


def run(
    cfg: ConfigManager,
    src_dir: Path | None = None,
    out_dir: Path | None = None,
) -> Path:
    """Walk ``src_dir`` and write Obsidian wiki to ``out_dir``; return ``out_dir``."""
    src_root = src_dir or (
        Path(__file__).resolve().parents[4]
        / str(cfg.get("graphify.src_dir", "src/roomba_lab"))
    )
    out_root = out_dir or (
        Path(__file__).resolve().parents[4]
        / str(cfg.get("graphify.output_dir", "docs/wiki"))
    )
    result = walk_source_tree(src_root, package_prefix="roomba_lab")
    emit(result, out_root)
    _logger.info("graphify wrote %d nodes + %d edges to %s",
                  len(result.nodes), len(result.edges), out_root)
    return out_root

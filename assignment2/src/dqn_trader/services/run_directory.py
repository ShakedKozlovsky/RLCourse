"""Per-training-run directory layout: snapshot config, checkpoints, metrics."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunDirectory:
    """A timestamped folder under ``results/`` for one training run."""

    root: Path

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def plots(self) -> Path:
        return self.root / "plots"

    @property
    def metrics_csv(self) -> Path:
        return self.root / "metrics.csv"

    def write_config_snapshot(self, config: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "config_snapshot.json").write_text(json.dumps(config, indent=2))

    def write_git_hash(self) -> None:
        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=self.root, stderr=subprocess.DEVNULL
            )
        except Exception:
            sha = b"unknown\n"
        (self.root / "git_hash.txt").write_bytes(sha)


def create_run(results_dir: Path, *, prefix: str = "run") -> RunDirectory:
    """Create ``results_dir/<prefix>_<unix_ts>/`` and its subfolders."""
    ts = int(time.time())
    root = results_dir / f"{prefix}_{ts}"
    root.mkdir(parents=True, exist_ok=True)
    (root / "checkpoints").mkdir(exist_ok=True)
    (root / "plots").mkdir(exist_ok=True)
    return RunDirectory(root=root)

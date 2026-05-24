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
        """Path to the checkpoints/ subdirectory."""
        return self.root / "checkpoints"

    @property
    def plots(self) -> Path:
        """Path to the plots/ subdirectory."""
        return self.root / "plots"

    @property
    def metrics_csv(self) -> Path:
        """Path to the per-episode metrics CSV."""
        return self.root / "metrics.csv"

    def write_config_snapshot(self, config: dict[str, Any]) -> None:
        """Write a JSON snapshot of the config used for this run."""
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "config_snapshot.json").write_text(json.dumps(config, indent=2))

    def write_git_hash(self) -> None:
        """Record the current git commit hash for reproducibility."""
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

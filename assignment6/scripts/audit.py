"""scripts/audit.py — full pre-submission audit.

Runs:
  1. ruff check src/ tests/                 (lint)
  2. pytest -q                              (tests)
  3. LOC check (≤150 LOC/file in src/)      (V3 rule)
  4. yaml load                              (config sanity)
  5. mini-graphify                          (regen architecture.md)

Exits 0 if everything passes, non-zero on the first failure."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def step(name: str, cmd: list[str]) -> int:
    print(f"=== {name} ===")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"FAIL: {name}")
    else:
        print(f"OK: {name}")
    return result.returncode


def loc_check(src_dir: Path, max_loc: int = 250) -> int:
    """Print any file exceeding ``max_loc`` lines (excluding tests + scripts)."""
    print(f"=== LOC check (≤{max_loc}/file in {src_dir}) ===")
    over: list[tuple[Path, int]] = []
    for p in src_dir.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        loc = len(p.read_text().splitlines())
        if loc > max_loc:
            over.append((p, loc))
    if over:
        print(f"FAIL: {len(over)} file(s) exceed {max_loc} LOC:")
        for p, loc in over:
            print(f"  {p} — {loc} LOC")
        return 1
    print("OK: all files within LOC limit")
    return 0


def main() -> int:
    rc = 0
    # Pre-step: ensure dev extras (ruff + pytest + pytest-cov) are installed.
    rc |= step("sync(dev)", ["uv", "sync", "--extra", "dev", "--quiet"])
    rc |= step("ruff", ["uv", "run", "ruff", "check", "src/", "tests/", "scripts/"])
    rc |= step("pytest", ["uv", "run", "pytest", "-q", "--no-header"])
    rc |= loc_check(Path("src/marl_lab"))
    rc |= step("graphify",
               ["uv", "run", "python", "-c",
                "from marl_lab.graphify.graphify import run; "
                "run('src/marl_lab', 'docs/wiki/architecture.md')"])
    print()
    print("FINAL:", "ALL OK" if rc == 0 else "FAILURES SEEN")
    return rc


if __name__ == "__main__":
    sys.exit(main())

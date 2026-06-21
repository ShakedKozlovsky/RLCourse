"""Meta-consistency tests that block the recurring layer-count drift pattern.

The TA flagged the same off-by-N drift 4 times across reviews (NEW7, NEW17, NEW18):
every time a new ``## Layer N`` entry was appended to ``docs/TODO.md``, the
README and EXEC_SUMMARY intro lines became stale by 1. Layer 32 ends the
chain by making the consistency a TEST. Adding a new layer without bumping
the intro number now fails CI."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _count_todo_layers() -> int:
    """Count the `## Layer ` headings in docs/TODO.md — single source of truth."""
    todo = (ROOT / "docs" / "TODO.md").read_text()
    return sum(1 for line in todo.splitlines() if line.startswith("## Layer "))


def _intro_layer_count(doc_path: Path) -> int:
    """Extract the integer N from a phrase like "**N layered task entries**".

    Raises if the phrase is absent or unparseable."""
    text = doc_path.read_text()
    match = re.search(r"\*\*(\d+) layered task entries\*\*", text)
    if not match:
        raise AssertionError(
            f"{doc_path.relative_to(ROOT)} does not contain a "
            '"**N layered task entries**" phrase — intro consistency check '
            "cannot verify it"
        )
    return int(match.group(1))


def test_readme_layer_count_matches_todo() -> None:
    todo_count = _count_todo_layers()
    readme_count = _intro_layer_count(ROOT / "README.md")
    assert readme_count == todo_count, (
        f"README.md intro says {readme_count} layered task entries; "
        f"docs/TODO.md has {todo_count}. Update one to match the other."
    )


def test_executive_summary_layer_count_matches_todo() -> None:
    todo_count = _count_todo_layers()
    exec_count = _intro_layer_count(ROOT / "docs" / "EXECUTIVE_SUMMARY.md")
    assert exec_count == todo_count, (
        f"EXECUTIVE_SUMMARY.md intro says {exec_count} layered task entries; "
        f"docs/TODO.md has {todo_count}. Update one to match the other."
    )


def test_todo_has_at_least_one_layer() -> None:
    """Sanity — TODO is never empty."""
    assert _count_todo_layers() >= 1

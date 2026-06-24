"""Convert notebooks/marl_walkthrough.py → executed HTML in docs/wiki/.

Usage: ``uv run python scripts/rebuild_notebook.py``

Three-step pipeline:
  1. jupytext: .py → .ipynb (preserves all `# %%` cell boundaries)
  2. nbconvert --execute: run every cell and capture outputs
  3. nbconvert --to html: render the executed notebook for in-browser viewing

The rendered HTML lands at ``docs/wiki/marl_walkthrough.html`` so a TA can
read the full walkthrough — code + outputs — without setting up a Python
environment."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_PY = ROOT / "notebooks" / "marl_walkthrough.py"
IPYNB = ROOT / "notebooks" / "marl_walkthrough.ipynb"
EXECUTED = ROOT / "notebooks" / "marl_walkthrough_executed.ipynb"
HTML_OUT_DIR = ROOT / "docs" / "wiki"
HTML_OUT = HTML_OUT_DIR / "marl_walkthrough.html"


def step(name: str, cmd: list[str]) -> int:
    print(f"=== {name} ===")
    rc = subprocess.run(cmd, check=False).returncode
    print(f"{'OK' if rc == 0 else 'FAIL'}: {name}")
    return rc


def main() -> int:
    HTML_OUT_DIR.mkdir(parents=True, exist_ok=True)
    rc = 0
    rc |= step("jupytext (.py → .ipynb)",
                ["uv", "run", "jupytext", "--to", "ipynb", str(SRC_PY)])
    rc |= step("nbconvert --execute",
                ["uv", "run", "jupyter", "nbconvert",
                 "--to", "notebook", "--execute", str(IPYNB),
                 "--output", EXECUTED.name,
                 "--ExecutePreprocessor.timeout=300"])
    rc |= step("nbconvert → HTML",
                ["uv", "run", "jupyter", "nbconvert", "--to", "html",
                 str(EXECUTED),
                 "--output-dir", str(HTML_OUT_DIR),
                 "--output", HTML_OUT.name])
    print()
    if rc == 0:
        print(f"HTML rendered to: {HTML_OUT}")
    return rc


if __name__ == "__main__":
    sys.exit(main())

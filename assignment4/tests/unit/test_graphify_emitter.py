"""Emitter — graph.json + module Markdown with Wikilinks."""

from __future__ import annotations

import json
from pathlib import Path

from proximal_lab.tools.graphify.emitter import emit
from proximal_lab.tools.graphify.walker import Edge, GraphifyResult, Node


def _result_with_one_module() -> GraphifyResult:
    return GraphifyResult(
        nodes=[
            Node(id="module:pkg.demo", type="module", module="pkg.demo",
                  path="demo.py", docstring_first_line="A demo."),
            Node(id="class:pkg.demo.Foo", type="class", module="pkg.demo",
                  path="demo.py", lineno=3, docstring_first_line="The Foo class."),
            Node(id="function:pkg.demo.bar", type="function", module="pkg.demo",
                  path="demo.py", lineno=10, docstring_first_line="Run bar."),
            Node(id="module:pkg.other", type="module", module="pkg.other",
                  path="other.py", docstring_first_line="Other."),
        ],
        edges=[
            Edge(src="module:pkg.demo", dst="class:pkg.demo.Foo", type="contains"),
            Edge(src="module:pkg.demo", dst="function:pkg.demo.bar", type="contains"),
            Edge(src="module:pkg.demo", dst="module:pkg.other", type="imports"),
        ],
    )


def test_graph_json_valid_schema(tmp_path: Path) -> None:
    emit(_result_with_one_module(), tmp_path)
    blob = json.loads((tmp_path / "graph.json").read_text())
    assert blob["version"] == "1.00"
    assert len(blob["nodes"]) == 4
    assert any(n["type"] == "module" and n["module"] == "pkg.demo"
                for n in blob["nodes"])
    assert any(e["type"] == "imports" for e in blob["edges"])


def test_module_markdown_has_wikilinks(tmp_path: Path) -> None:
    emit(_result_with_one_module(), tmp_path)
    md = (tmp_path / "pkg_demo.md").read_text()
    assert "# pkg.demo" in md
    assert "The Foo class." in md
    assert "Run bar." in md
    assert "[[pkg_other|pkg.other]]" in md


def test_index_lists_all_modules(tmp_path: Path) -> None:
    emit(_result_with_one_module(), tmp_path)
    idx = (tmp_path / "_index.md").read_text()
    assert "[[pkg_demo|pkg.demo]]" in idx
    assert "[[pkg_other|pkg.other]]" in idx


def test_empty_module_emits_no_docstring(tmp_path: Path) -> None:
    result = GraphifyResult(
        nodes=[Node(id="module:pkg.x", type="module", module="pkg.x",
                     path="x.py", docstring_first_line="")],
        edges=[],
    )
    emit(result, tmp_path)
    md = (tmp_path / "pkg_x.md").read_text()
    assert "(no docstring)" in md

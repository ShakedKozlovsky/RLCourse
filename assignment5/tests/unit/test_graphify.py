"""Layer 10 — graphify walker + emitter on a synthetic fixture."""

from __future__ import annotations

import json
from pathlib import Path

from roomba_lab.tools.graphify.emitter import emit
from roomba_lab.tools.graphify.walker import walk_source_tree


def _make_fixture(root: Path) -> None:
    (root / "a.py").write_text(
        '"""mod a."""\nfrom b import f\nclass A:\n    """ClassA."""\n\n'
        'def public_fn():\n    """do."""\n    return 1\n'
    )
    (root / "b.py").write_text(
        '"""mod b."""\nfrom a import A\n\ndef f():\n    """fn f."""\n    return 0\n'
    )
    (root / "__init__.py").write_text("")


def test_walker_counts_nodes_and_edges(tmp_path: Path) -> None:
    _make_fixture(tmp_path)
    result = walk_source_tree(tmp_path, package_prefix="syn")
    modules = [n for n in result.nodes if n.type == "module"]
    classes = [n for n in result.nodes if n.type == "class"]
    funcs = [n for n in result.nodes if n.type == "function"]
    assert len(modules) == 2
    assert len(classes) == 1
    assert len(funcs) == 2
    import_edges = [e for e in result.edges if e.type == "imports"]
    contain_edges = [e for e in result.edges if e.type == "contains"]
    assert len(import_edges) == 2
    assert len(contain_edges) == 3


def test_emitter_writes_graph_json_and_markdown(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _make_fixture(src)
    result = walk_source_tree(src, package_prefix="syn")
    out = tmp_path / "wiki"
    emit(result, out)
    assert (out / "graph.json").exists()
    assert (out / "_index.md").exists()
    payload = json.loads((out / "graph.json").read_text())
    assert payload["version"] == "1.00"
    assert len(payload["nodes"]) == len(result.nodes)
    a_md = out / "syn_a.md"
    b_md = out / "syn_b.md"
    assert a_md.exists() and b_md.exists()
    assert "## Imports" in a_md.read_text()
    assert "[[b" in a_md.read_text()

"""Emit ``graph.json`` + one Markdown file per module — Obsidian-compatible.

Each module Markdown file contains:
    - YAML frontmatter with path, module, tags
    - Module docstring as ``## Summary``
    - List of classes + functions defined in the module
    - **Wikilinks** to imported modules: ``[[module-id]]``
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from roomba_lab.tools.graphify.walker import Edge, GraphifyResult, Node


def emit(result: GraphifyResult, out_dir: Path) -> None:
    """Write ``graph.json`` + one ``.md`` per module to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_graph_json(result, out_dir)
    _write_module_markdowns(result, out_dir)
    _write_index(result, out_dir)


def _write_graph_json(result: GraphifyResult, out_dir: Path) -> None:
    payload = {
        "version": "1.00",
        "nodes": [asdict(n) for n in result.nodes],
        "edges": [asdict(e) for e in result.edges],
    }
    (out_dir / "graph.json").write_text(json.dumps(payload, indent=2))


def _slug(module: str) -> str:
    return module.replace(".", "_")


def _write_module_markdowns(result: GraphifyResult, out_dir: Path) -> None:
    by_module = _index_nodes_by_module(result.nodes)
    edges_by_src = _index_edges_by_src(result.edges)
    for mod_node in (n for n in result.nodes if n.type == "module"):
        slug = _slug(mod_node.module)
        lines = _module_markdown(mod_node, by_module, edges_by_src)
        (out_dir / f"{slug}.md").write_text("\n".join(lines))


def _module_markdown(
    mod_node: Node,
    by_module: dict[str, list[Node]],
    edges_by_src: dict[str, list[Edge]],
) -> list[str]:
    lines = [
        "---",
        f"path: {mod_node.path}",
        f"module: {mod_node.module}",
        "tags: [type-module]",
        "---",
        "",
        f"# {mod_node.module}",
        "",
        "## Summary",
        mod_node.docstring_first_line or "(no docstring)",
        "",
    ]
    members = by_module.get(mod_node.module, [])
    classes = [n for n in members if n.type == "class"]
    funcs = [n for n in members if n.type == "function"]
    if classes:
        lines.extend(["## Classes", ""])
        for c in classes:
            lines.append(f"- `{c.module}.{c.id.split('.')[-1]}` — "
                          f"{c.docstring_first_line or 'no docstring'}")
        lines.append("")
    if funcs:
        lines.extend(["## Public functions", ""])
        for f in funcs:
            lines.append(f"- `{f.module}.{f.id.split('.')[-1]}` — "
                          f"{f.docstring_first_line or 'no docstring'}")
        lines.append("")
    imported = [e for e in edges_by_src.get(mod_node.id, []) if e.type == "imports"]
    if imported:
        lines.extend(["## Imports", ""])
        for e in imported:
            target_module = e.dst.removeprefix("module:")
            lines.append(f"- [[{_slug(target_module)}|{target_module}]]")
    return lines


def _write_index(result: GraphifyResult, out_dir: Path) -> None:
    lines = ["# Project Wiki Index", ""]
    for n in result.nodes:
        if n.type == "module":
            lines.append(f"- [[{_slug(n.module)}|{n.module}]] — "
                          f"{n.docstring_first_line or '(no docstring)'}")
    (out_dir / "_index.md").write_text("\n".join(lines))


def _index_nodes_by_module(nodes: list[Node]) -> dict[str, list[Node]]:
    out: dict[str, list[Node]] = {}
    for n in nodes:
        if n.type != "module":
            out.setdefault(n.module, []).append(n)
    return out


def _index_edges_by_src(edges: list[Edge]) -> dict[str, list[Edge]]:
    out: dict[str, list[Edge]] = {}
    for e in edges:
        out.setdefault(e.src, []).append(e)
    return out

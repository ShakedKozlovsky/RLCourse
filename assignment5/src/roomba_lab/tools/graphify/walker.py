"""AST walker — extract modules, classes, public functions, and import edges.

Output is a pair ``(nodes, edges)`` consumable by ``emitter.py``.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Node:
    """One symbol in the knowledge graph."""

    id: str
    type: str           # "module" | "class" | "function"
    module: str
    path: str
    lineno: int = 0
    docstring_first_line: str = ""


@dataclass(frozen=True)
class Edge:
    """One relation between two nodes."""

    src: str
    dst: str
    type: str           # "imports" | "contains"


@dataclass
class GraphifyResult:
    """Container for walker output."""

    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)


def walk_source_tree(src_root: Path, package_prefix: str = "") -> GraphifyResult:
    """Walk every ``.py`` file under ``src_root`` (excluding __init__.py)."""
    result = GraphifyResult()
    root = src_root.resolve()
    for py in sorted(root.rglob("*.py")):
        if py.name == "__init__.py":
            continue
        module = _module_id(py, root, package_prefix)
        _walk_one(py, module, result)
    return result


def _module_id(py: Path, root: Path, prefix: str) -> str:
    rel = py.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if prefix:
        parts = [prefix, *parts]
    return ".".join(parts)


def _walk_one(py: Path, module: str, result: GraphifyResult) -> None:
    try:
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
    except SyntaxError:
        return
    docstring = (ast.get_docstring(tree) or "").splitlines()[:1]
    module_doc = docstring[0] if docstring else ""
    mod_node = Node(
        id=f"module:{module}", type="module", module=module,
        path=str(py), lineno=0, docstring_first_line=module_doc,
    )
    result.nodes.append(mod_node)
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            result.nodes.append(Node(
                id=f"class:{module}.{node.name}", type="class", module=module,
                path=str(py), lineno=node.lineno,
                docstring_first_line=_first_doc_line(node),
            ))
            result.edges.append(Edge(
                src=mod_node.id, dst=f"class:{module}.{node.name}", type="contains",
            ))
        elif isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            result.nodes.append(Node(
                id=f"function:{module}.{node.name}", type="function", module=module,
                path=str(py), lineno=node.lineno,
                docstring_first_line=_first_doc_line(node),
            ))
            result.edges.append(Edge(
                src=mod_node.id, dst=f"function:{module}.{node.name}", type="contains",
            ))
        elif isinstance(node, ast.ImportFrom) and node.module:
            target_mod = node.module
            result.edges.append(Edge(
                src=mod_node.id, dst=f"module:{target_mod}", type="imports",
            ))


def _first_doc_line(node: ast.AST) -> str:
    doc = ast.get_docstring(node) or ""
    return doc.splitlines()[0] if doc else ""

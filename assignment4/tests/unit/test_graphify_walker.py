"""AST walker — extract modules, classes, public functions, import edges."""

from __future__ import annotations

from pathlib import Path

from proximal_lab.tools.graphify.walker import walk_source_tree


def test_simple_module_extracted(tmp_path: Path) -> None:
    """A synthetic 1-module fixture with 1 class + 1 function → 3 nodes."""
    (tmp_path / "demo.py").write_text(
        '"""Demo module docstring."""\n'
        "\n"
        "class Foo:\n"
        '    """Foo class docstring."""\n'
        "    pass\n"
        "\n"
        "def bar():\n"
        '    """Bar function docstring."""\n'
        "    return 1\n"
    )
    result = walk_source_tree(tmp_path, package_prefix="pkg")
    assert any(n.type == "module" for n in result.nodes)
    assert any(n.type == "class" and "Foo" in n.id for n in result.nodes)
    assert any(n.type == "function" and "bar" in n.id for n in result.nodes)
    docstrings = {n.docstring_first_line for n in result.nodes if n.type == "module"}
    assert "Demo module docstring." in docstrings


def test_imports_become_edges(tmp_path: Path) -> None:
    """``from pkg.X import Y`` → edge ``module:pkg.A → module:pkg.X``."""
    (tmp_path / "a.py").write_text(
        "from pkg.helpers import do_thing\n"
        "\n"
        "def use_it():\n"
        "    return do_thing()\n"
    )
    result = walk_source_tree(tmp_path, package_prefix="pkg")
    edges = [e for e in result.edges if e.type == "imports"]
    assert any("pkg.helpers" in e.dst for e in edges)


def test_underscore_functions_excluded(tmp_path: Path) -> None:
    """Private ``_helper`` is not emitted as a node."""
    (tmp_path / "demo.py").write_text(
        "def public_fn():\n"
        "    pass\n"
        "\n"
        "def _private_fn():\n"
        "    pass\n"
    )
    result = walk_source_tree(tmp_path, package_prefix="pkg")
    fn_names = [n.id for n in result.nodes if n.type == "function"]
    assert any("public_fn" in n for n in fn_names)
    assert not any("_private_fn" in n for n in fn_names)


def test_init_py_excluded(tmp_path: Path) -> None:
    (tmp_path / "__init__.py").write_text('"""Should not appear."""\n')
    (tmp_path / "real.py").write_text('"""Should appear."""\n')
    result = walk_source_tree(tmp_path, package_prefix="pkg")
    modules = [n.module for n in result.nodes if n.type == "module"]
    assert any("real" in m for m in modules)
    assert not any("__init__" in m for m in modules)


def test_syntax_error_file_skipped(tmp_path: Path) -> None:
    (tmp_path / "broken.py").write_text("def bad(:\n    pass\n")
    (tmp_path / "good.py").write_text('"""Fine module."""\n')
    result = walk_source_tree(tmp_path, package_prefix="pkg")
    modules = [n.module for n in result.nodes if n.type == "module"]
    assert any("good" in m for m in modules)


def test_walks_real_project(tmp_path: Path) -> None:
    """Smoke: walking the actual proximal_lab source tree produces edges."""
    src = Path(__file__).resolve().parents[2] / "src" / "proximal_lab"
    result = walk_source_tree(src, package_prefix="proximal_lab")
    assert len(result.nodes) > 10
    assert any(e.type == "imports" for e in result.edges)

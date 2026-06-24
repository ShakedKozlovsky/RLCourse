"""Layer 20 — Mini-Graphify tests."""

from __future__ import annotations

from pathlib import Path

from marl_lab.graphify.graphify import (
    format_markdown,
    run,
    summarise_module,
    walk_source_tree,
)


def test_summarise_module_extracts_docstring(tmp_path: Path) -> None:
    p = tmp_path / "x.py"
    p.write_text('"""hello world"""\n\ndef foo() -> int: return 1\n')
    s = summarise_module(p)
    assert s.docstring == "hello world"
    assert "foo" in s.public_symbols


def test_summarise_module_skips_private_symbols(tmp_path: Path) -> None:
    p = tmp_path / "x.py"
    p.write_text('"""x"""\n\ndef _private() -> None: ...\nclass Public: ...\n')
    s = summarise_module(p)
    assert "_private" not in s.public_symbols
    assert "Public" in s.public_symbols


def test_summarise_module_handles_syntax_errors(tmp_path: Path) -> None:
    p = tmp_path / "broken.py"
    p.write_text("def foo(:")    # syntax error
    s = summarise_module(p)
    assert "syntax error" in s.docstring


def test_walk_source_tree_picks_up_marl_lab() -> None:
    src = Path("src/marl_lab")
    files = walk_source_tree(src)
    assert len(files) > 10               # we have many modules
    assert any("dec_pomdp.py" in str(f.path) for f in files)
    assert any("qmix_mixer.py" in str(f.path) for f in files)


def test_format_markdown_groups_by_pillar() -> None:
    src = Path("src/marl_lab")
    files = walk_source_tree(src)
    md = format_markdown(files, src)
    # Pillar headers
    assert "## model/" in md or "## services/" in md
    assert "auto-generated" in md


def test_run_writes_output_file(tmp_path: Path) -> None:
    out = tmp_path / "arch.md"
    path = run(src_dir="src/marl_lab", output_path=out)
    assert path == out
    assert out.exists()
    content = out.read_text()
    assert "marl_lab" in content
    assert "auto-generated" in content

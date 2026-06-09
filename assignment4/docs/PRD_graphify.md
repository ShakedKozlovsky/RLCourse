# PRD — Mini-Graphify Tool (Active Knowledge Architecture realisation)

## What and why

The *Active Knowledge Architecture* methodology document positions the project as a **two-layer system**:

| Layer | Owner | Purpose |
|---|---|---|
| **Obsidian** | Human | Navigation layer — Wikilinks, ontology, manual curation |
| **Graphify** | AI / automation | Computation layer — AST parsing, semantic inference, Knowledge Graph generation |

The lecturer's workflow (slide 4 of the methodology doc):
```
1. Raw Folder        → 2. Pipeline (Graphify)    → 3. Wiki Folder           → 4. Obsidian Vault
   (papers, code,       (AST + LLM semantic         (graph.json + .md         (the human-readable
    screenshots,         inference → knowledge       per-module with           knowledge base, opens
    tweets)              graph)                      Wikilinks)                in Obsidian)
```

**Mini-Graphify** is our implementation of step 2 — focused, no LLM dependency, walks the project's *own* Python AST. It is a **methodology differentiator**, not a serious replacement for the full Graphify.

## What it does (concretely)

1. Walk `src/proximal_lab/` with `ast.parse`.
2. Extract:
   - **Module nodes** — one per `.py` file (excl. `__init__.py`).
   - **Class nodes** — class definitions inside each module.
   - **Function nodes** — public (non-underscore-prefixed) functions.
   - **Import edges** — `from proximal_lab.X import Y` becomes `module(X) → module(Y)`.
   - **Optional call-graph edges** — method-level calls within a module (off by default for simplicity).
3. Emit:
   - `docs/wiki/graph.json` — `{nodes: [...], edges: [...]}` with `type`, `id`, `module`, `lineno` fields.
   - `docs/wiki/<module-name>.md` — one Markdown file per module containing:
     - YAML frontmatter: `path`, `module`, `tags: [type-module]`
     - First-line docstring as `## Summary`
     - Class definitions listed with first-line docstrings
     - Public functions listed with signatures
     - **Wikilinks** to imported modules: `[[shared_config]]`, `[[environment_mujoco_env]]`, etc.

## Why a custom mini-tool, not just `pyreverse` or `pydeps`?

- **Pedagogical**: implementing it ourselves proves we understand the *Active Knowledge Architecture* idea, not just the tooling.
- **Obsidian-tailored**: existing tools produce DOT or SVG; we produce Markdown with Wikilinks, which is Obsidian's native graph format.
- **Tiny scope**: ~200 LOC across walker + emitter + runner. Won't dwarf the RL code.

## Output structure

```
docs/wiki/
├── graph.json                       # the full knowledge graph
├── _index.md                        # entry point with Wikilinks to top-level modules
├── shared_config.md
├── shared_logger.md
├── shared_seed.md
├── shared_types.md
├── environment_mujoco_env.md
├── environment_vector_env.md
├── model_actor.md
├── model_critic.md
├── model_actor_critic_network.md
├── services_gae.md
├── services_ppo_service.md
├── services_rollout_buffer.md
├── services_evaluation_service.md
├── sdk_sdk.md
└── tools_graphify_walker.md         # the tool documents itself, neatly meta
```

Opening `docs/wiki/` in Obsidian shows the project as a graph (Obsidian's graph view aggregates all `[[wikilinks]]`). The lecturer's diagram (page 1 of the methodology PDF) becomes runnable.

## graph.json schema

```json
{
  "version": "1.00",
  "generated_at": "2026-06-09T21:47:00Z",
  "source_root": "src/proximal_lab",
  "nodes": [
    {"id": "module:services.gae",        "type": "module",   "path": "src/proximal_lab/services/gae.py",        "lines": 60},
    {"id": "function:services.gae.compute_gae", "type": "function", "module": "services.gae", "lineno": 12},
    {"id": "class:services.ppo_service.PPOService", "type": "class", "module": "services.ppo_service", "lineno": 34}
  ],
  "edges": [
    {"src": "module:services.ppo_service", "dst": "module:services.gae",        "type": "imports"},
    {"src": "module:services.ppo_service", "dst": "module:services.rollout_buffer", "type": "imports"}
  ]
}
```

## Acceptance criteria

- `test_walker.py::test_simple_module_extracted` — synthetic 1-file fixture with 1 class + 1 function produces 3 nodes.
- `test_walker.py::test_imports_become_edges` — `from proximal_lab.X import Y` produces an `imports` edge.
- `test_walker.py::test_underscore_functions_excluded` — `_helper` does not appear as a node.
- `test_emitter.py::test_graph_json_valid` — output JSON matches the schema above.
- `test_emitter.py::test_module_markdown_has_wikilinks` — module Markdown contains `[[other-module]]` for every import.
- `test_runner.py::test_cli_runs_on_real_src` — `proximal-lab graphify` runs against `src/proximal_lab` and exits 0.

## Where this lives

- `src/proximal_lab/tools/graphify/walker.py` — `ast.NodeVisitor` subclass extracting nodes/edges ≤ 100 LOC.
- `src/proximal_lab/tools/graphify/emitter.py` — JSON + Markdown writer ≤ 80 LOC.
- `src/proximal_lab/tools/graphify/runner.py` — CLI entry ≤ 30 LOC.

## Caveats

- We do **not** do semantic LLM inference like the real Graphify does. Edges are based on explicit imports only.
- We do **not** parse PDFs, screenshots, or other non-code files. The methodology document positions Graphify as multi-modal; mini-Graphify is code-only.
- Cycles in the import graph are reported, not flagged as errors. They're valid Python.
- We do not handle relative imports beyond one level (`from .X import Y` works; `from ..X import Y` is rare in this codebase and unsupported).

## How the methodology pays off

After Layer 9 the user can:

```bash
# Build the wiki
uv run proximal-lab graphify

# Open it in Obsidian (any platform)
open docs/wiki/_index.md

# Or just browse on GitHub — the .md files render natively with Wikilinks shown as raw text
```

The Layer 15 README will embed a screenshot of the Obsidian graph view as one of its figures — closing the loop between the methodology document, the code, and the final visualisation.

## Sources

- *Active Knowledge Architecture* methodology PDF (Dr. Yoram Segal, May 2026; NotebookLM).
- Python `ast` module documentation — https://docs.python.org/3/library/ast.html.
- Obsidian Wikilink syntax — `[[target]]`, `[[target|display text]]`.

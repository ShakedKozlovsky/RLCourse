---
path: /home/corsight/src/RL_Course/assignment4/src/proximal_lab/tools/graphify/walker.py
module: proximal_lab.tools.graphify.walker
tags: [type-module]
---

# proximal_lab.tools.graphify.walker

## Summary
AST walker — extract modules, classes, public functions, and import edges.

## Classes

- `proximal_lab.tools.graphify.walker.Node` — One symbol in the knowledge graph.
- `proximal_lab.tools.graphify.walker.Edge` — One relation between two nodes.
- `proximal_lab.tools.graphify.walker.GraphifyResult` — Container for walker output.

## Public functions

- `proximal_lab.tools.graphify.walker.walk_source_tree` — Walk every ``.py`` file under ``src_root`` (excluding __init__.py).

## Imports

- [[__future__|__future__]]
- [[dataclasses|dataclasses]]
- [[pathlib|pathlib]]
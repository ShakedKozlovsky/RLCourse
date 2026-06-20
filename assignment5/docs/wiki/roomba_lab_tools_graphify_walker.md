---
path: /home/corsight/src/RL_Course/assignment5/src/roomba_lab/tools/graphify/walker.py
module: roomba_lab.tools.graphify.walker
tags: [type-module]
---

# roomba_lab.tools.graphify.walker

## Summary
AST walker — extract modules, classes, public functions, and import edges.

## Classes

- `roomba_lab.tools.graphify.walker.Node` — One symbol in the knowledge graph.
- `roomba_lab.tools.graphify.walker.Edge` — One relation between two nodes.
- `roomba_lab.tools.graphify.walker.GraphifyResult` — Container for walker output.

## Public functions

- `roomba_lab.tools.graphify.walker.walk_source_tree` — Walk every ``.py`` file under ``src_root`` (excluding __init__.py).

## Imports

- [[__future__|__future__]]
- [[dataclasses|dataclasses]]
- [[pathlib|pathlib]]
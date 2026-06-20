---
path: /home/corsight/src/RL_Course/assignment5/src/roomba_lab/interface/cli/main.py
module: roomba_lab.interface.cli.main
tags: [type-module]
---

# roomba_lab.interface.cli.main

## Summary
Top-level Click entry point. Bulky command bodies live in `commands.py`.

## Public functions

- `roomba_lab.interface.cli.main.cli` — roomba-lab — custom DDPG cleaning-robot laboratory.
- `roomba_lab.interface.cli.main.train` — Train a DDPG agent.
- `roomba_lab.interface.cli.main.evaluate` — Evaluate a saved checkpoint deterministically.
- `roomba_lab.interface.cli.main.download_data` — Confirm the 10-map sample shipped in the repo.
- `roomba_lab.interface.cli.main.record_gif` — Record a GIF of one cleaning episode.
- `roomba_lab.interface.cli.main.sweep` — Run an empirical sweep over noise σ, τ, or target-network strategy.
- `roomba_lab.interface.cli.main.graphify` — Emit the docs/wiki/ Obsidian Vault from src/roomba_lab via AST walk.
- `roomba_lab.interface.cli.main.gui` — Launch the PyQt6 GUI (training + visualisation tabs).

## Imports

- [[__future__|__future__]]
- [[pathlib|pathlib]]
- [[roomba_lab_interface_cli|roomba_lab.interface.cli]]
- [[roomba_lab_shared_config|roomba_lab.shared.config]]
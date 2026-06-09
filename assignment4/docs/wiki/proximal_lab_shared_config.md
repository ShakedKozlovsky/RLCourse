---
path: /home/corsight/src/RL_Course/assignment4/src/proximal_lab/shared/config.py
module: proximal_lab.shared.config
tags: [type-module]
---

# proximal_lab.shared.config

## Summary
Configuration loader. Single source of truth — no hardcoded values in code.

## Classes

- `proximal_lab.shared.config.ConfigError` — Raised on missing / malformed / version-mismatched config.
- `proximal_lab.shared.config.ConfigManager` — Loads JSON config and exposes it via dotted access.

## Imports

- [[__future__|__future__]]
- [[collections_abc|collections.abc]]
- [[pathlib|pathlib]]
- [[typing|typing]]
- [[proximal_lab_shared_version|proximal_lab.shared.version]]
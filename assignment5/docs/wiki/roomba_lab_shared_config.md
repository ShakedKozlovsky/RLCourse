---
path: /home/corsight/src/RL_Course/assignment5/src/roomba_lab/shared/config.py
module: roomba_lab.shared.config
tags: [type-module]
---

# roomba_lab.shared.config

## Summary
Configuration loader for roomba-lab. Single source of truth — no hardcoded values

## Classes

- `roomba_lab.shared.config.ConfigError` — Raised on missing / malformed / version-mismatched config.
- `roomba_lab.shared.config.ConfigManager` — Loads JSON config and exposes it via dotted access.

## Imports

- [[__future__|__future__]]
- [[collections_abc|collections.abc]]
- [[pathlib|pathlib]]
- [[typing|typing]]
- [[roomba_lab_shared_version|roomba_lab.shared.version]]
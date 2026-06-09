---
path: /home/corsight/src/RL_Course/assignment4/src/proximal_lab/services/comparison_service.py
module: proximal_lab.services.comparison_service
tags: [type-module]
---

# proximal_lab.services.comparison_service

## Summary
Cross-config comparison — aggregate seeds with mean ± 95 % CI.

## Classes

- `proximal_lab.services.comparison_service.CellSummary` — One config × seeds aggregate — used for sweep cells + cross-env cells.
- `proximal_lab.services.comparison_service.ComparisonReport` — A full sweep / comparison result for a configurable list of cells.
- `proximal_lab.services.comparison_service.ComparisonService` — Build a :class:`ComparisonReport` from a dict ``{cell_name: [per-seed finals]}``.

## Public functions

- `proximal_lab.services.comparison_service.aggregate_with_ci` — Mean ± 95 % normal-approx CI of a list of per-seed final rewards.

## Imports

- [[__future__|__future__]]
- [[dataclasses|dataclasses]]
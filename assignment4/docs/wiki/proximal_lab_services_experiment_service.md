---
path: /home/corsight/src/RL_Course/assignment4/src/proximal_lab/services/experiment_service.py
module: proximal_lab.services.experiment_service
tags: [type-module]
---

# proximal_lab.services.experiment_service

## Summary
Empirical sweeps: λ-sweep, γ-sweep, clip-ε sweep with multi-seed CIs.

## Classes

- `proximal_lab.services.experiment_service.ExperimentService` — Run a sweep over one hyperparameter, aggregating across seeds.

## Public functions

- `proximal_lab.services.experiment_service.aggregate_seeds` — Mean ± std + final reward summary; exposed for tests.

## Imports

- [[__future__|__future__]]
- [[copy|copy]]
- [[pathlib|pathlib]]
- [[tempfile|tempfile]]
- [[proximal_lab_sdk_sdk|proximal_lab.sdk.sdk]]
- [[proximal_lab_services_comparison_service|proximal_lab.services.comparison_service]]
- [[proximal_lab_shared_logger|proximal_lab.shared.logger]]
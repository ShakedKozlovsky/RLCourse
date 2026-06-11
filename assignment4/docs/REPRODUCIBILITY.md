# Reproducibility Statement

> Every plot, number, and table cited in the README + executive summary maps to a specific script in this repository. This document is the auditable trail.

## Hardware & software baseline

| Item | Value |
|---|---|
| OS | Linux 6.8 (Ubuntu 24.04) |
| Python | 3.12 (managed via `uv python install 3.12`) |
| Package manager | `uv` 0.11.2 — locked in [`uv.lock`](../uv.lock) |
| PyTorch | 2.12 (CPU build) |
| MuJoCo | 3.x via `pip install mujoco` |
| Gymnasium | 0.29 with `gymnasium[mujoco]` |
| GPU usage | None — all numbers are CPU-only |

Reproducing on a different OS may need `MUJOCO_GL=osmesa` or `MUJOCO_GL=glfw` instead of `egl`.

## Bit-identical contract

The reproducibility guarantee tested by [`tests/integration/test_reproducibility.py`](../tests/integration/test_reproducibility.py):

> **Same `configs/setup.json` + same seed → bit-identical per-iteration reward and KL trajectories.**

This holds across machines for CPU-only runs, modulo MuJoCo's documented non-determinism across CPU architectures over very long horizons. Tested on the build machine for runs up to ~50k timesteps.

Note: running two SDKs in parallel before training both is **not** supported — the second sees an advanced RNG state. Construct, train, repeat.

## Plot-by-plot reproduction

| Asset | Generator | Inputs |
|---|---|---|
| [`assets/plots/lambda_sweep.png`](../assets/plots/lambda_sweep.png) | [`scripts/generate_plots.py`](../scripts/generate_plots.py) | [`results/sweeps/lambda.json`](../results/sweeps/lambda.json) |
| [`assets/plots/lambda_sweep_multiseed.png`](../assets/plots/lambda_sweep_multiseed.png) **headline** | `scripts/generate_plots.py` | [`results/sweeps/lambda_multiseed.json`](../results/sweeps/lambda_multiseed.json) |
| [`assets/plots/gamma_sweep.png`](../assets/plots/gamma_sweep.png) | `scripts/generate_plots.py` | [`results/sweeps/gamma.json`](../results/sweeps/gamma.json) |
| [`assets/plots/clip_eps_sweep.png`](../assets/plots/clip_eps_sweep.png) | `scripts/generate_plots.py` | [`results/sweeps/clip_eps.json`](../results/sweeps/clip_eps.json) |
| [`assets/plots/cross_env_comparison.png`](../assets/plots/cross_env_comparison.png) | `scripts/generate_plots.py` | [`results/layer11/cross_env.json`](../results/layer11/cross_env.json) |
| [`assets/plots/clipped_surrogate.png`](../assets/plots/clipped_surrogate.png) | [`scripts/plot_clipped_surrogate.py`](../scripts/plot_clipped_surrogate.py) | analytic — no inputs |
| [`assets/plots/gae_ablation.png`](../assets/plots/gae_ablation.png) | [`scripts/run_gae_ablation.py`](../scripts/run_gae_ablation.py) | [`results/layer16_gae_ablation/gae_ablation_raw.json`](../results/layer16_gae_ablation/gae_ablation_raw.json) |
| [`assets/diagrams/architecture.png`](../assets/diagrams/architecture.png) | [`scripts/plot_architecture.py`](../scripts/plot_architecture.py) | [`docs/wiki/graph.json`](wiki/graph.json) |
| [`assets/gifs/walker2d_trained.gif`](../assets/gifs/walker2d_trained.gif) | [`scripts/render_policy_gif.py`](../scripts/render_policy_gif.py) | [`saved_models/Walker2d-v5.pt`](../saved_models/Walker2d-v5.pt) |
| [`assets/gifs/halfcheetah_trained.gif`](../assets/gifs/halfcheetah_trained.gif) | `scripts/render_policy_gif.py` | [`saved_models/HalfCheetah-v5.pt`](../saved_models/HalfCheetah-v5.pt) |
| [`assets/gui/tab_*.png`](../assets/gui/) | [`scripts/capture_gui.py`](../scripts/capture_gui.py) | live SDK |

## Number-by-number reproduction

The multi-seed λ-sweep headline table is the headline empirical claim. Reproduce:

```bash
cd assignment4
uv sync --extra dev
uv run python scripts/run_lambda_multiseed.py  # ~3 min CPU
uv run python scripts/generate_plots.py
```

Reads `configs/setup.json:experiments.lambda_sweep` for the λ grid (default `[0.0, 0.5, 0.9, 0.95, 0.99, 1.0]`). Produces `results/sweeps/lambda_multiseed.json` + `assets/plots/lambda_sweep_multiseed.png` deterministically given the seed.

## Wall-clock budget for a full reproduction

| Item | Approx |
|---|---|
| `uv sync --extra dev` (first time) | ~3 min (large PyTorch download) |
| Test suite | ~10 s |
| Layer-10 single-seed sweeps (λ, γ, ε) | ~3 min total |
| Layer-13 multi-seed λ-sweep (3×6 cells × 15k steps) | ~3 min |
| Layer-11 cross-env (2 envs × 30k steps) | ~50 s |
| Layer-16 GAE ablation (3λ × 3 seeds × 15k steps) | ~90 s |
| Layer-9 graphify | <1 s |
| Render policy GIFs | ~30 s |
| All sweep plots | <2 s |
| **Total full rebuild** | **~12 min on a modern CPU** |

## What is *not* reproducible

- **Absolute MuJoCo reward magnitudes across very different CPUs** — Gymnasium MuJoCo wrappers depend on platform-specific FP behaviour for very long horizons. Single-iteration outputs are identical; 1M-step trajectories may diverge after ~100k steps.
- **GUI screenshots on different DPI / display backends** — visual layout may differ; the underlying data does not.
- **CI runner wall-clock** — varies with GitHub-hosted-runner load; the *numeric outputs* are stable.

## Audit checklist for the grader

1. ✅ `uv sync --extra dev` succeeds — `pyproject.toml` + `uv.lock` work together.
2. ✅ `uv run pytest tests/ -q` shows 120 tests passing.
3. ✅ `uv run ruff check src/ tests/` returns 0 errors.
4. ✅ `uv run pytest --cov=src/proximal_lab --cov-report=term` shows ≥ 85 %.
5. ✅ All scripts in `scripts/` run with `uv run python scripts/<name>.py`.
6. ✅ Every plot in `assets/plots/` is referenced in the README with a working relative link.
7. ✅ Notebook `notebooks/proximal_lab_walkthrough.ipynb` opens on GitHub with all cells showing pre-executed outputs.

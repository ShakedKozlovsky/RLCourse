# Executive Summary — fitness-rl (1-pager for the grader)

> Built layer-by-layer (Layers 0–15) on `main`. **235 tests · 97.5 % coverage · ruff clean.**

## What was built

A complete policy-gradient RL system that learns daily workout recommendations (`PUSH / PULL / LEGS / CARDIO / REST`) over an LSTM-learned world model from the Kaggle "600 K+ Fitness Exercise & Workout Program" dataset. Three algorithms compared end-to-end: **REINFORCE → A2C → PPO** (PPO is beyond-spec). User-facing CLI + 5-tab PyQt6 GUI + Jupyter notebook walkthrough.

## Assignment compliance

| Part | Requirement | Delivered |
|---|---|---|
| A | RL formulation (MDP + state/action/reward) | [§ 4](../README.md#4-state-action-reward-env-part-a--b) |
| B | Kaggle data → 84-day synthetic trainee trajectory | [§ 3](../README.md#3-data-pipeline-part-b) |
| C | LSTM world model + supervised training | [§ 5](../README.md#5-lstm-world-model-part-c--with-baselines) + 3.2× lower MSE than persistence |
| D | REINFORCE with reward-to-go + baseline | [§ 6](../README.md#6-reinforce-part-d) |
| E | A2C with TD-error advantage | [§ 7](../README.md#7-a2c-part-e) |
| F | 5 reflection answers + Action Masking (excellence) | [§ 12](../README.md#12-five-reflection-answers-part-f) + [`docs/PRD_action_masking.md`](PRD_action_masking.md) |
| F+ | PPO (beyond-spec extension) | [`docs/PRD_ppo.md`](PRD_ppo.md) + [§ 9.1](../README.md#91-multi-seed-comparison-with-95-ci-audit-3--18) |

## Headline empirical results (3 seeds × 300 episodes, post-reward-fix)

| Algorithm | Final-30 % mean reward | 95 % CI | vs best baseline |
|---|---|---|---|
| **REINFORCE** | **8.20** | ± 2.23 | +5.0× over round-robin |
| A2C | 5.24 | ± 1.73 | +3.2× over round-robin |
| PPO | 4.06 | ± 3.64 | +2.5× over round-robin |
| round-robin baseline | 1.64 | — | (reference) |
| Kaggle program | **−1.47** | — | overload-penalised |

All three trained agents beat all three baseline policies. The actual Kaggle program scores *negative* under the (corrected) reward — proof that the reward function meaningfully penalises bad scheduling.

## Engineering polish

- **15 layers**, one commit per layer, `Layer N: <summary>` convention throughout
- **235 tests** (unit + integration + headless-Qt GUI smoke), **97.5 % branch coverage** (gate 85 %)
- `ruff check src/ tests/` returns 0
- **Every source file ≤ 150 LOC** except `sdk/sdk.py` (154 LOC, documented exception)
- No magic numbers — all hyperparameters in `configs/setup.json`
- Version sync across `shared/version.py` / `pyproject.toml` / `configs/setup.json`
- Reproducibility test: same seed → bit-identical training histories

## Adversarial-audit response (Layers 11–15)

After Layer 10 I asked Claude to play the role of a critical professor; it surfaced 20 weaknesses. Layers 11–15 address every one of them:

- **Audit #1 — LSTM unvalidated** → persistence + linear baselines + multi-step rollout MSE
- **Audit #2 — A2C collapse** → entropy sweep proves the reward was mis-specified; **Layer 15 reward fix** (REST = 0 gain) resolves it
- **Audit #3 + #18 — single-seed claims** → 3–5 seed multi-seed runs with 95 % CI
- **Audit #4 — no baselines** → random / round-robin / Kaggle-program reference policies
- **Audit #7 — chain not demonstrated** → REINFORCE → +baseline → +advantage → A2C → PPO empirical chain
- **Audit #9 — reward unvalidated** → entropy sweep + multi-seed evidence, fix applied in Layer 15
- **Audit #16 — no diagram** → `assets/diagrams/architecture.png`
- **Audit #20 — coverage gaps** → 96.6 % → **97.5 %**

Full mapping of all 20 findings to fixes: [README § 9.7 audit-findings status table](../README.md#audit-findings-status).

## Where to look first

1. [`README.md`](../README.md) — full project doc with plots and reflection answers
2. [`notebooks/fitness_rl_walkthrough.ipynb`](../notebooks/fitness_rl_walkthrough.ipynb) — 6-cell guided tour
3. [`assets/plots/three_algo_curves.png`](../assets/plots/three_algo_curves.png) — the headline 3-algo result
4. [`results/layer15/full_budget_multiseed.json`](../results/layer15/full_budget_multiseed.json) — raw numbers

## Honest acknowledgements

- The Kaggle dataset is workout *programs*, not physiological *outcomes*. The reward function operationalises three constructs (gain, overload, imbalance) — domain-expert proxies, not biological measurements.
- The trained agents pick PUSH/CARDIO heavily; the reward design favours simple high-volume actions. A schedule-quality bonus (mutual information with a balanced training plan) is documented as future work.
- PPO scores lowest of the three algorithms here despite its theoretical advantages. With 300 episodes the PPO buffer is small (28 steps); a longer run with multi-step batches would likely flip this. Documented honestly rather than tuned away.

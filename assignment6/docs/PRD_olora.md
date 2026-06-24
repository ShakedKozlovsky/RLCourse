# PRD — OLoRA (Orthonormal Low-Rank Adaptation)

> Per-mechanism PRD. L10 § 6 + spec § 5.2 mention OLoRA as the recommended PEFT for pre-trained backbones in unstable RL training. [Büyükakyüz 2024, arXiv:2406.01775]

## 1. Why OLoRA in MARL

RL training is **inherently unstable** — gradients can spike, target networks lag, policies oscillate. Standard LoRA initialises the low-rank A factor as a random Gaussian, which produces noisy outputs at step 0. In stable supervised fine-tuning that's fine; in RL the noise feeds straight into the policy gradient and amplifies oscillation.

**OLoRA** replaces the random Gaussian init with a **QR-decomposition-based orthonormal init**. The result: at step 0 the OLoRA output is *deterministic and well-conditioned*; training is dramatically more stable.

## 2. The math (from the OLoRA paper)

For a pre-trained linear layer `W_pre ∈ R^{d_out × d_in}`, OLoRA reparametrises:

$$W = W_{pre} + \alpha \cdot B \cdot A^\top, \quad A \in \mathbb{R}^{d_{in} \times r}, \quad B \in \mathbb{R}^{d_{out} \times r}$$

with **orthonormal init** for `A`:

$$A_{init} = Q \quad \text{where} \quad W_{pre}^\top = Q R \;\; (\text{QR decomposition})$$

`B` starts at zero, so `B · Aᵀ = 0` at step 0 — the wrapped layer behaves identically to `W_pre` initially. Training updates `B` (and optionally `A`); orthonormality of `A` preserves the conditioning of the update.

## 3. Where it lives

| Item | File |
|---|---|
| `OLoRAAdapter(base_layer, rank)` class | `src/marl_lab/model/olora.py` |
| Helper: `wrap_with_olora(model, rank)` | same file |
| Toggle: `configs/setup.yaml::marl.use_olora` + `marl.olora_rank` | config |

## 4. When to use it

| Scenario | Use OLoRA? |
|---|---|
| Train from scratch on 5x5 grid | No — base model is small enough to fully train |
| Warm-start from a 3x3-trained policy + scale to 5x5 (curriculum) | **Yes** — OLoRA preserves the small-grid knowledge while adapting |
| Warm-start from a generic backbone (e.g. CNN encoder) | **Yes** — OLoRA is what makes the warm-start stable |
| Need to scale to 6x6 / 8x8 in the final project | **Yes** — pre-train at 5x5, OLoRA-adapt at 6x6 |

## 5. Test plan

| Test | Pass criterion |
|---|---|
| Orthonormality of init A | `A.T @ A ≈ I_rank` (within float32 tolerance) |
| Zero-perturbation at init | `WrappedLayer(x) == W_pre @ x` (since B = 0) |
| Gradient flows to B (not pre-trained weights) | `B.grad` exists, `W_pre.grad` is None |
| Rank preserved | `rank(B @ A.T) == r` post-training |
| Parameter count | `(d_in + d_out) · r` instead of `d_in · d_out` |

## 6. Acceptance criteria

1. `OLoRAAdapter` is a pure `nn.Module` (no global state).
2. `wrap_with_olora(model)` walks the model tree and wraps every `nn.Linear` (configurable via target-pattern arg).
3. Math tests cover all 5 items in § 5 above.
4. A documented ablation in the experiments sweep: `use_olora=true` vs `false` on a curriculum (2x2 → 3x3) training run.

## 7. Non-goals

- LoRA-fa (frozen-A variant) — defer to final project.
- Multi-layer LoRA / convolution layers (`marl_lab` uses only Linear + GRU, both handleable).
- AdaLoRA (rank-adaptive) — out of scope.

## 8. Citation

Büyükakyüz, K. *OLoRA: Orthonormal Low-Rank Adaptation of Large Language Models*. arXiv:2406.01775, 2024. https://arxiv.org/abs/2406.01775

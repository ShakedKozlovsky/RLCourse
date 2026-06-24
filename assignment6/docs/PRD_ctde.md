# PRD — CTDE + VDN + QMIX + IGM (the headline algorithm machinery)

> Per-mechanism PRD. L10 § 4 is the contract. **The heart of the MARL contribution.**

## 1. The CTDE paradigm

**Centralised Training, Decentralised Execution** = the dominant approach to escape Dec-POMDP non-stationarity (L10 § 3).

- **Training**: a central learner sees `s` (global), `ō` (joint local obs), `ā` (joint actions), `r̄` (per-agent or joint rewards).
- **Execution**: each agent uses ONLY its own `Qᵢ(oᵢ)` — no global state access.

The value decomposition is what links the two: the centralised `Q_tot` is a function of per-agent `Qᵢ` outputs, and the IGM principle guarantees the centralised argmax factorises into per-agent argmaxes.

## 2. VDN (Sunehag et al. AAMAS 2018)

The simplest functional form:

$$Q_{tot}(\bar{o}, \bar{a}) = \sum_{i=1}^{n} Q_i(o_i, a_i)$$

- IGM automatically satisfied (each `Qᵢ` is monotone in itself).
- Trivially decomposable: train against centralised target; at execution each agent argmaxes its own Qᵢ.
- Limited expressiveness — can't represent value functions where one agent's marginal value depends on another agent's action.

## 3. QMIX (Rashid et al. ICML 2018)

A non-linear monotonic generalisation:

$$Q_{tot}(\bar{o}, \bar{a}, s) = f_\theta(Q_1, Q_2, \dots, Q_n; s) \quad \text{with} \quad \frac{\partial f_\theta}{\partial Q_i} \geq 0 \;\; \forall i$$

- `f_θ` is a small neural net (the "mixer") whose weights are produced by a **hypernetwork** conditioned on the global state `s`.
- Monotonicity enforced by `|W|` (absolute-value) parametrisation on the mixer weights.
- **Strictly more expressive** than VDN; **strictly less expressive** than fully unconstrained mixers (which would violate IGM).

```
            ┌─────────────┐
   Q₁ ──┐   │             │   (positive weights via |W|)
   Q₂ ──┼──>│   MIXER     │──> Q_tot
   ...  │   │  (2-layer   │
   Qₙ ──┘   │   MLP)      │
            └──────▲──────┘
                   │ weights come from
            ┌──────┴──────┐
            │ HYPERNET    │
            │ (state → W) │
            └──────▲──────┘
                   │
                global state s
```

## 4. IGM principle (Individual-Global-Max)

$$\arg\max_{\bar{a}} Q_{tot}(s, \bar{a}) = \left( \arg\max_{a_1} Q_1(o_1, a_1), \dots, \arg\max_{a_n} Q_n(o_n, a_n) \right)$$

**Why IGM matters**: at execution, agent i only sees `oᵢ`. The agent argmaxes its own `Qᵢ`. IGM guarantees that doing so is *also* the global argmax of `Q_tot`. Without IGM, you'd need a centralised executor at runtime — defeating the whole point of CTDE.

### When IGM holds

| Mixer | IGM? | Why |
|---|---|---|
| Sum (VDN) | ✓ | Each Qᵢ is independent in the sum |
| Monotonic mixer (QMIX) | ✓ | All ∂Q_tot/∂Qᵢ ≥ 0 → argmax preserved |
| QPLEX (Wang 2021) | ✓ (extended) | Duplex dueling decomposition |
| Arbitrary mixer | ✗ | Cross-terms can invert per-agent argmaxes |

## 5. The TD target (L10 equation 4 generalised)

For each agent's update step we use the centralised target:

$$y = \bar{r} + \gamma \cdot \max_{\bar{a}'} Q_{tot}^{target}(s', \bar{a}') = \bar{r} + \gamma \cdot Q_{tot}^{target}(s', \arg\max_{a'_i} Q_i^{target}(o'_i, a'_i))$$

i.e. each agent's target `Qᵢ` selects its own argmax (using IGM), then the centralised target mixer combines them.

## 6. Critical analysis — non-stationarity + IQL failure

**IQL** (Independent Q-Learning) ignores the multi-agent aspect entirely — each agent treats the others as part of the environment. From agent i's perspective:

$$y_i = r_i + \gamma \cdot \max_{a'_i} Q_i^{target}(o'_i, a'_i)$$

When the other agent updates its policy, the effective environment shifts. The IQL update assumes stationary dynamics; this is the **non-stationarity problem** [Foerster 2018; Amato 2024]. Empirically, IQL converges in adversarial domains slowly or to Nash-equilibrium-far policies.

QMIX/VDN escape this by sharing centralised gradient information at training time (the `s`-conditioned mixer + the centralised target).

## 7. Test plan

| Test | Pass criterion |
|---|---|
| VDN sum identity | `mixer([q1, q2]) == q1 + q2` exactly |
| QMIX monotonicity | finite-difference: `∂Q_tot/∂Qᵢ ≥ 0` for 100 random `(Q, s)` |
| QMIX state-dependence | same Qᵢ + different s → different Q_tot |
| QMIX-via-equal-weights ≈ VDN | when hypernet outputs equal weights, QMIX ≈ VDN sum (modulo bias) |
| One-step gradient flow | `loss.backward()` populates `Qᵢ.grad` AND `Mixer.grad`; targets have no grad |
| Target-network Polyak | τ=0.005 produces target_drift > 0 after one update |

## 8. Hyperparameter justification (spec § 5.5 Item 3)

| Param | Default | Why |
|---|---|---|
| `marl.algorithm` | qmix | The headline; falls back to vdn or iql via config swap |
| `marl.gamma` | 0.99 | Standard discount; sub-game horizon ≈ 25 |
| `marl.tau` | 0.005 | Polyak target update — matches A5 / Lillicrap |
| `marl.mixer_lr` | 1e-3 | Same as critic LR; mixer is part of the critic |
| `marl.use_rnn` | true | Partial-obs → GRU hidden state encodes recent history |
| `marl.rnn_hidden_size` | 64 | Mid-size memory for ≤ 25-step horizons |
| `marl.hidden_sizes` | [128, 128] | Per-agent Q backbone — large enough for the 2D Manhattan-radius input |
| `marl.batch_size` | 64 | RNN sequence batch — smaller than DDPG because each batch has T-step BPTT |
| `marl.replay_capacity` | 100 000 transitions | 5 000 episodes × ~20 steps |
| `marl.warmup_steps` | 500 | Fill the buffer with diverse episodes |

## 9. Acceptance criteria

1. All three updaters (`iql_update`, `vdn_update`, `qmix_update`) share the `Updater` protocol.
2. Each updater is unit-tested on a 3-test gradient battery.
3. The QMIX hypernetwork is finite-difference-verified to be monotonic.
4. The IGM principle is documented as a one-line assertion in the code (`assert (∂Q_tot/∂Qᵢ ≥ 0).all()` in the test).
5. Empirical comparison plot (Layer 21) shows QMIX > VDN > IQL on at least the 3×3 + 4×4 grids.

## 10. Non-goals

- QPLEX (mentioned as extension point, not implemented — saved for the final project)
- Weighted-QMIX (extension point only)
- COMA / CounterFactual-MAPG (mentioned in spec § 7.2 — implementable as a 100-LOC siblings module, deferred to final-project layer)

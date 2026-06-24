# PRD — Independent Q-Learning (IQL) baseline

> Per-mechanism PRD. L10 § 3 + spec § 7.2 ("Baseline") + Foerster 2018 + Amato 2024 [4]. The empirical comparison anchor for the CTDE-vs-naïve argument.

## 1. Why IQL is the right baseline

Spec § 7.2 explicitly demands:

> *"Comparison: must compare the autonomous-learning approach against IQL, while noting IQL's limitations in non-stationary environments."*

The TA-graded contrast is: **does QMIX/VDN beat IQL?** If yes (which the literature predicts), the CTDE machinery is paying its complexity cost; if no, we have explaining to do.

## 2. The IQL algorithm

Each agent runs its own Q-learning update, treating the others as part of the environment:

$$y_i = r_i + \gamma \max_{a'_i} Q_i^{target}(o'_i, a'_i)$$

$$L_i = \mathbb{E}\left[(Q_i(o_i, a_i) - y_i)^2\right]$$

**Key property**: NO mixer. Each agent's loss depends ONLY on its own observations, actions, rewards, and target Q. The agents are coupled implicitly through the environment dynamics (the other agent's actions change `o'`), but there is no coordinated gradient.

## 3. What goes wrong (the non-stationarity story)

From agent i's perspective:

$$P(o'_i \mid o_i, a_i) = \sum_{a_{-i}} \pi_{-i}(a_{-i} \mid o_{-i}) \cdot T(s' \mid s, a_i, a_{-i})$$

The other agent's policy `π_{-i}` is **changing during training**. So the effective transition kernel `P(o'_i | o_i, a_i)` is **non-stationary**.

Q-learning's convergence proof requires stationary transitions. IQL violates the assumption. Empirically:

- IQL converges very slowly (or not at all) in adversarial domains.
- Even when it converges, it often lands far from Nash equilibrium.
- Increasing the discount γ amplifies the problem (longer credit assignment magnifies the drift).

## 4. Implementation

Lives in `src/marl_lab/services/iql_update.py`. Structurally identical to `qmix_update.py` minus the mixer:

```python
def apply_iql_update(q_nets, target_q_nets, batch, hp):
    # Per agent INDEPENDENTLY:
    for agent_id in batch.agents:
        b = batch.for_agent(agent_id)
        with torch.no_grad():
            q_next = target_q_nets[agent_id](b.next_obs).max(dim=-1).values
            y = b.reward + hp.gamma * (1 - b.done) * q_next
        q_pred = q_nets[agent_id](b.obs)[b.action]
        loss = F.mse_loss(q_pred, y)
        loss.backward()
        # Per-agent optimiser step
    # Per-agent Polyak (independent)
    for agent_id in batch.agents:
        polyak_update(target_q_nets[agent_id], q_nets[agent_id], hp.tau)
```

**Key differences from QMIX**:

| | IQL | QMIX |
|---|---|---|
| Mixer | None | Hypernet-conditioned monotonic |
| Target | `r_i + γ max_a' Q_i(o'_i, a')` | `r + γ Q_tot(s', argmax)` |
| Uses global state `s` during training | No (only `o_i`) | Yes |
| Updates each Q in isolation | Yes | Joint via centralised loss |
| Communicates between agents | No | Via shared mixer gradient |

## 5. Test plan

| Test | Pass criterion |
|---|---|
| Per-agent independence | `loss_i.backward()` does NOT populate `q_nets[j].grad` for `j ≠ i` |
| Gradient flows in each agent | `q_nets[i].grad` is non-None for each `i` |
| Per-agent target Polyak | `target_q_nets[i]` moves toward `q_nets[i]` independently |
| Algorithm switch via config | `marl.algorithm = "iql"` selects this updater; same train loop as QMIX/VDN |
| Same hyperparams | IQL uses same γ, τ, lr, batch as QMIX (apples-to-apples) |

## 6. Empirical comparison plan (Layer 21)

| Setup | Cells | Seeds | Expected outcome |
|---|---|---|---|
| 3×3 grid | IQL, VDN, QMIX | 3 | QMIX ≥ VDN > IQL (per Sunehag 2018 / Rashid 2018) |
| 4×4 grid | IQL, VDN, QMIX | 3 | Gap widens |
| 5×5 grid | IQL, VDN, QMIX | 3 | Gap widens further |

Plot: bar chart with t-distribution 95% CIs (carried from A5 v1.20+). The plot is the headline of reflection-Q1 ("how does CTDE beat IQL?").

## 7. Acceptance criteria

1. `IqlUpdater` shares the `Updater` protocol with QMIX/VDN — fully swappable.
2. Per-agent independence verified in unit tests.
3. The Layer 21 algorithm sweep produces a JSON + plot that shows the predicted ranking on at least one grid size.
4. Reflection-Q1 answer cites the Layer 21 plot.

## 8. Honest caveats

- IQL **can** work surprisingly well in some domains (small grids, short horizons, dense rewards). Cops-and-Robbers at 2×2 / 3×3 may not show a big gap.
- Our 25-move sub-game horizon is short → non-stationarity has less time to bite → IQL may approach VDN/QMIX. Document this in the analysis if it happens.

## 9. Non-goals

- IQL with experience replay sharing (would partially fix non-stationarity — saved for final project)
- Multi-objective IQL — out of scope
- Fingerprint-augmented IQL [Foerster 2017] — out of scope

## 10. Citations

- Foerster et al., *Counterfactual Multi-Agent Policy Gradients*, AAAI 2018 — discusses IQL's failure modes.
- Amato, *A First Introduction to Cooperative MARL*, arXiv:2405.06161, 2024.
- Sunehag et al., *Value-Decomposition Networks*, AAMAS 2018 — empirical IQL vs VDN comparison.

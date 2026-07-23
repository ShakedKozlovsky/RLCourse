# PROOFS — Formal IGM derivations for VDN / QMIX / QPLEX

Companion to README § 7.2. Each section proves the Individual–Global–Max (IGM) property for one mixer family in this codebase, with explicit reference to the implementing module and the empirical test that verifies the math.

The **IGM principle** (Son et al. ICML 2019; restated in Wang 2021 § 2) is the equivalence between centralised joint argmax and decentralised per-agent argmax:

$$\operatorname*{arg\,max}_{\bar{a}} Q_{\text{tot}}(\boldsymbol\tau, \bar{a})
\;=\;\bigl(\operatorname*{arg\,max}_{a_1} Q_1(\tau_1, a_1),\,\ldots,\,
\operatorname*{arg\,max}_{a_n} Q_n(\tau_n, a_n)\bigr).$$

A sufficient (and very common) condition is **per-agent monotonicity** of the mixer:

$$\frac{\partial Q_{\text{tot}}}{\partial Q_i}(\boldsymbol\tau, \bar{a}) \,\geq\, 0
\quad\text{for every agent } i.$$

When the mixer is monotone in every $Q_i$, raising any single agent's $Q_i$ cannot lower $Q_{\text{tot}}$, so the joint argmax co-varies with each per-agent argmax — i.e. IGM holds.

---

## 1. VDN — additive sum trivially satisfies IGM

**Mixer** (`src/marl_lab/model/vdn_mixer.py`):

$$Q_{\text{tot}}(\boldsymbol\tau, \bar{a}) \;=\; \sum_{i=1}^{N} Q_i(\tau_i, a_i).$$

**Claim.** $\partial Q_{\text{tot}} / \partial Q_i = 1 > 0$ for every $i$, so IGM holds.

**Proof.** Differentiating the sum:

$$\frac{\partial}{\partial Q_i} \sum_{j=1}^{N} Q_j \;=\; \sum_{j=1}^{N} \frac{\partial Q_j}{\partial Q_i} \;=\; \mathbb{1}[j=i]\Big|_{j=i} \;=\; 1.$$

So the partial derivative is the constant $1$, which is strictly positive. ∎

**Test:** `tests/unit/test_mixers.py::test_vdn_sum_identity_2_agents` — directly checks `Q_tot([3.0, 4.0]) == 7.0`. Plus `test_vdn_sum_identity_n_agents` for $N=5$.

**Caveat.** VDN's representational expressiveness is *strictly less* than QMIX or QPLEX: it cannot model any non-additive interaction between agents (no cross-terms, no state conditioning). The sum identity is both its strength (trivial IGM) and its limit.

---

## 2. QMIX — monotonic-mixer IGM via $|\,\cdot\,|$ parametrisation

**Mixer** (`src/marl_lab/model/qmix_mixer.py::QMIXMixer.forward` — the `torch.abs()` calls at lines 86 and 90 enforce the non-negativity that carries the whole proof):

The QMIX mixer is a two-layer feedforward network whose **weights** $W_1, W_2$ are produced by a hypernet conditioned on the global state $s$:

$$h \;=\; \text{ELU}\bigl( |W_1(s)| \cdot \mathbf{Q} + b_1(s) \bigr),$$
$$Q_{\text{tot}}(\boldsymbol\tau, \bar{a}, s) \;=\; |W_2(s)| \cdot h \;+\; b_2(s),$$

where $\mathbf{Q} \in \mathbb{R}^{N}$ is the stacked per-agent values $(Q_1, \ldots, Q_N)$, the $|\cdot|$ is elementwise absolute value, ELU is the Exponential Linear Unit (Clevert 2015), and biases $b_1(s), b_2(s)$ are **unconstrained** (can take either sign).

**Claim.** $\partial Q_{\text{tot}} / \partial Q_i \geq 0$ for every $i$ and every input $(\boldsymbol\tau, \bar{a}, s)$.

**Proof.** Apply the chain rule layer by layer.

- *Layer 2.* Let $\widetilde{W}_2 = |W_2(s)| \in \mathbb{R}^{1 \times \dim h}$. Then $\partial Q_{\text{tot}} / \partial h = \widetilde{W}_2^\top$, and every entry of $\widetilde{W}_2$ is non-negative by construction.

- *Layer 1.* Let $\widetilde{W}_1 = |W_1(s)| \in \mathbb{R}^{\dim h \times N}$. The pre-activation is $z = \widetilde{W}_1 \mathbf{Q} + b_1(s)$ so $\partial z / \partial Q_i = (\widetilde{W}_1)_{:,i} \geq 0$. The activation derivative is $\partial h / \partial z = \text{ELU}'(z)$, which is **non-negative everywhere**:

$$\text{ELU}'(z) \;=\; \begin{cases} 1, & z > 0 \\ \alpha\, e^{z}, & z \leq 0\end{cases} \;\geq\; 0
\qquad(\alpha > 0).$$

- *Composition.* By the chain rule,

$$\frac{\partial Q_{\text{tot}}}{\partial Q_i} \;=\; \widetilde{W}_2 \cdot \text{diag}(\text{ELU}'(z)) \cdot (\widetilde{W}_1)_{:,i}.$$

Each of the three factors is elementwise non-negative; their product (as a sum of products of non-negatives) is non-negative. ∎

**Test:** `tests/unit/test_mixers.py::test_qmix_monotonicity_finite_difference` — 100 random $(\mathbf{Q}, s)$ probes with `n_agents=2`, plus 50 probes with `n_agents=5` (`test_qmix_monotonicity_finite_difference_n5`), each verifying `torch.autograd.grad(Q_tot, Q)[i] >= 0` for every $i$.

**Why $|W|$ and not e.g. $W^2$.** Both $|W|$ and $W^2$ produce non-negative outputs but $W^2$ would also send gradients to zero whenever $W = 0$, creating dead units. The $|W|$ choice is the one used in the original Rashid 2018 paper and our implementation.

**Caveat.** QMIX's monotonicity guarantees IGM but *restricts* the representable family of $Q_{\text{tot}}$ to monotonically-factorisable functions. Concretely, QMIX cannot represent landscapes where the joint argmax disagrees with the marginal argmax (e.g. XOR-like coordination tasks). See QPLEX § 3 below for the strict generalisation.

---

## 3. QPLEX — IGM by construction via dueling decomposition

**Mixer** (`src/marl_lab/model/qplex_mixer.py`):

QPLEX (Wang et al. ICLR 2021, arXiv:2008.01062) decomposes $Q_{\text{tot}}$ into a global value head plus per-agent dueling advantages:

$$V_i(\tau_i) \;\stackrel{\text{def}}{=}\; \max_{a_i'} Q_i(\tau_i, a_i'),$$
$$A_i(\tau_i, a_i) \;\stackrel{\text{def}}{=}\; Q_i(\tau_i, a_i) - V_i(\tau_i) \;\leq\; 0,$$
$$Q_{\text{tot}}(\boldsymbol\tau, \bar{a}, s) \;=\; V_{\text{tot}}(s) \;+\; \sum_{i=1}^{N} \lambda_i(s)\,A_i(\tau_i, a_i),$$

with $\lambda_i(s) > 0$ enforced via $\lambda_i(s) = |\text{hypernet}_\lambda(s)|_i$ and $V_{\text{tot}}(s)$ a hypernet output of unconstrained sign.

**Claim 1 (IGM).** For every $i$,
$\operatorname*{arg\,max}_{a_i} Q_{\text{tot}}(\boldsymbol\tau, \bar{a}, s)
= \operatorname*{arg\,max}_{a_i} Q_i(\tau_i, a_i)$ .

**Proof.** Holding $s$ and $\bar{a}_{-i}$ (the other agents' actions) fixed, only the $i$-th advantage depends on $a_i$:

$$\frac{\partial Q_{\text{tot}}}{\partial Q_i(\tau_i, a_i)}
\;=\; \lambda_i(s) \cdot \underbrace{\frac{\partial A_i}{\partial Q_i}}_{= 1} \;=\; \lambda_i(s) \;>\; 0.$$

So $Q_{\text{tot}}$ is strictly monotone increasing in $Q_i(\tau_i, a_i)$, which means $\arg\max_{a_i} Q_{\text{tot}} = \arg\max_{a_i} Q_i$. The argument holds for every agent independently, so the joint argmax decomposes. ∎

**Claim 2 (strict expressiveness gain over QMIX).** There exists a state $s$ and a setting of QPLEX hypernet weights such that $Q_{\text{tot}}(s) < 0$ even when every $Q_i \geq 0$. (No such configuration exists for QMIX with bias $b_2(s) \geq 0$.)

**Proof.** Let $V_{\text{tot}}(s) = -3$ (achievable because the V-head is sign-unconstrained), every $Q_i = V_i = 1$ (so every $A_i = 0$), every $\lambda_i(s) > 0$. Then

$$Q_{\text{tot}} = V_{\text{tot}}(s) + \sum_i \lambda_i \cdot 0 = -3 < 0.$$ ∎

**Test:** `tests/unit/test_qplex.py::test_qplex_lambda_positivity_via_autograd` — 80 random $(\mathbf{Q}, V, s)$ probes verifying $\partial Q_{\text{tot}} / \partial Q_i > 0$ for $N = 3$. Plus `test_qplex_more_expressive_than_qmix` — drives $Q_{\text{tot}}$ to $-3$ via 100 SGD steps with all-positive $Q_i$, demonstrating Claim 2 empirically.

**Reduction at the argmax.** When every agent plays the greedy action ($Q_i = V_i$), every $A_i = 0$ and $Q_{\text{tot}} = V_{\text{tot}}(s)$. Tested by `test_qplex_at_argmax_reduces_to_v_tot`.

---

## 4. Bernstein 2002 — why exact Dec-POMDP solving is intractable

The factorisation methods above (VDN, QMIX, QPLEX) are not just *convenient* — they're a response to a deep complexity result.

**Theorem (Bernstein, Givan, Immerman, Zilberstein, 2002).** *The problem of solving a finite-horizon Dec-POMDP optimally is **NEXP-complete***. The proof reduces the *TILING* problem to Dec-POMDP planning and shows the resulting instances require non-deterministic exponential time in the worst case, even for two agents on a small state space.

**Consequence for this codebase.** Any tractable algorithm for our 5×5 cops-and-robbers task must be an *approximation*: there is no polynomial-time exact solver unless `P = NEXP` (which contradicts the time-hierarchy theorem). The CTDE family is one specific tractable approximation:

| Approach | Complexity (per training step) | Cost we pay |
|---|---|---|
| Exact Dec-POMDP | NEXP-complete | Intractable beyond ~10 states |
| Centralised joint-action Q | $O(\|A\|^N)$ per update | Action explosion as $N$ grows |
| Independent Q-learning (IQL) | $O(\|A\| \cdot N)$ | Non-stationarity (no convergence guarantee) |
| **VDN / QMIX / QPLEX (this codebase)** | $O(\|A\| \cdot N + \text{mixer})$ | Restricted factorisation (mitigated by QPLEX) |

The VDN / QMIX / QPLEX trio buys polynomial-time decentralised execution at the price of **representational restrictions** on $Q_{\text{tot}}$. We characterise those restrictions explicitly in § 1-3 above and verify them empirically.

**Where the spec's POSG framing changes the picture.** Bernstein's NEXP-completeness is for cooperative Dec-POMDPs with a *shared* reward. The cops-and-robbers task is technically a POSG — the cop and thief have opposite reward signals. The complexity for general POSG solving is **NEXP**$^{\text{NP}}$ (Hansen, Bernstein, Zilberstein 2004) — strictly harder. Our averaged-reward CTDE compromise (`(r_cop + r_thief) / 2` in `services/qmix_update.py` line 94) is a pragmatic choice that recovers the Dec-POMDP machinery; the trade-off is documented in [`FAILURE_MODES.md`](FAILURE_MODES.md) § 1 and empirically confirmed as harmful on this task by the v1.14 ELO tournament (FAILURE_MODES § 8): all three algorithms that go through this averaging (QMIX / VDN / QPLEX) rank BELOW uniform-random. **VDN is affected too even though its mixer is a pure sum** — its update path calls `apply_qmix_update` (`services/vdn_update.py` line 32) which averages the rewards before backprop.

**Why the bibliography ranks Bernstein 2002 first.** The cited paper (spec § 10 ref [1]) is the *reason* the field developed CTDE in the first place. Implementing CTDE without engaging with this complexity result would be implementing a remedy without understanding the disease. The reduction from TILING (a classic NEXP-complete problem) to Dec-POMDP planning establishes that no asymptotic improvement is possible by clever algorithm design alone — the only paths forward are (a) restrict the representable family (VDN / QMIX / QPLEX), (b) sample-based planning (Monte Carlo Tree Search), or (c) accept approximate solutions. We pursue path (a).

## 5. Summary table — IGM family of this codebase

| Mixer | $\partial Q_{\text{tot}} / \partial Q_i$ | Representable family | Module | Test |
|---|---|---|---|---|
| VDN  | $= 1$       | additive only           | `model/vdn_mixer.py` | `test_vdn_sum_identity_*` |
| QMIX | $\geq 0$    | monotone-factorisable   | `model/qmix_mixer.py` | `test_qmix_monotonicity_finite_difference[_n5]` |
| QPLEX| $> 0$       | full IGM-respecting     | `model/qplex_mixer.py` | `test_qplex_lambda_positivity_via_autograd` |

The trainer (`services/marl_trainer.py`) is mixer-agnostic — switching algorithms is a one-line config change (`algo="qmix" | "vdn" | "qplex" | "iql"`). IQL is the no-mixer baseline included for the non-stationarity comparison required by spec § 7.2.

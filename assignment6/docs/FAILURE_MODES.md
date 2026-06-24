# FAILURE_MODES.md — known limitations + honest disclosure

This file documents what `marl_lab` **doesn't** do well, why, and what would
fix it. Anti-hallucination by design — the TA should never have to ask
"does X work?" because the answer is here.

## 1. POSG framing vs Dec-POMDP machinery (ADR-002)

The cops-and-robbers game is technically a **POSG** (Partially Observable
Stochastic Game): each agent has its own reward function (the cop wants to
catch; the thief wants to escape). The CTDE training machinery we use —
VDN, QMIX, IGM — was designed for cooperative **Dec-POMDP** where all agents
share a single team reward.

We bridge this gap by treating the per-step reward as the average of the
two agent rewards (in `services/qmix_update.py`, the `joint_reward` line).
This is a known practical approximation; it works empirically because the
cop and thief learn opposing policies through SELF-PLAY, but the IGM
guarantee technically breaks under per-agent reward divergence.

**What would fix it:** A full POSG learner (e.g. MADDPG with two independent
centralised critics, or Nash-Q in tabular settings). That's beyond the
scope of A6's bonus assignment.

## 2. GRU hidden state is reset between MCP calls

The MCP servers store one GRU hidden tensor per connection (`_hidden`).
However the protocol doesn't carry a session id, so simultaneous clients
share state. For the local 6-sub-game flow this is fine (we call
`reset_hidden()` at the start of each sub-game), but a multi-tenant
deployment would need per-session tracking.

**What would fix it:** Add a session_id to `SelectActionRequest` and key
the hidden state by it.

## 3. Limited exploration in 5×5 grid

ε-greedy with linear decay over 50k steps is sufficient for 3×3 and 4×4
grids, but on 5×5 the cop sometimes converges to a local optimum where it
stays in a corner and waits for the thief to wander in. A more sophisticated
exploration scheme (e.g. count-based curiosity, or randomised network
distillation) would help.

**What would fix it:** Swap `noise/epsilon_greedy.py` for an RND or
intrinsic-curiosity exploration module. The `LinearEpsilonSchedule` is
already pluggable.

## 4. No model-based learning

The library is pure model-free. For sample efficiency on tiny grids
(where the transition function is simple enough to learn), a model-based
variant (Dyna-style) would converge in 10× fewer episodes. Out of scope.

## 5. Cloud deploy is "soft"

`cloud/prefect.py` falls back to local if `PREFECT_API_KEY` is missing.
This is the right choice for grading (avoids credential requirements) but
means the spec's § 5.3 "cloud deploy" line is satisfied with a single
remote invocation, not a always-on workflow.

**What would fix it:** Add `cloud/aws_batch.py` or similar always-on
backend. Out of scope.

## 6. Gmail App Password is a known weak credential

Per Google's policy, App Passwords are being phased out in favor of OAuth2.
Our `AppPasswordStrategy` works today but won't in late 2026. The
`OAuthStrategy` is the long-term path — we wire it but don't run the OAuth
dance in the smoke tests.

**What would fix it:** A one-shot `gmail-oauth-bootstrap` CLI command to
walk the user through the OAuth flow on first run. Could be added without
breaking changes.

## 7. Test coverage gaps

- The FastMCP HTTP transport (`cop_server.py:main`, `thief_server.py:main`)
  is excluded from coverage because spawning a real server in tests would
  require port management and SIGTERM coordination. The `BaseMCPServer`
  logic is tested in isolation, which covers the security-critical path.
- The Tkinter widget layer is excluded — headless CI can't render. The
  `GameGuiCore` + `board_renderer` cover the testable paths.
- OAuth + App Password actual network calls are not tested — only the
  arg-validation paths are.

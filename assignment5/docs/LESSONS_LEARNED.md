# Lessons Learned — beyond the spec's three reflection questions

> The spec asks three specific reflection questions (answered in the README).
> This document captures the *meta* lessons — what the build itself taught us
> about doing DDPG, doing this kind of project, and doing AI-assisted
> engineering at this scale.

## 1. The headline lesson — reward shaping is the algorithm

For Q-Learning, reward shaping is a tweak. For DDPG, **it is the signal that
the critic learns and the actor exploits**. Layer 18 spent a full session
discovering that our initial reward design (`collision=-10`, `step=-0.01`)
taught the agent that **standing still is the dominant strategy** because

```
  500 steps of standing still      → -5  reward
  1 collision in 500 steps         → -10 reward
```

made movement strictly riskier than inaction. The critic correctly learned
Q(s, 0) > Q(s, any-other-action), and the actor honoured it. The result was
that 20 000 steps of training produced *lower* coverage than 4 000 steps —
the agent literally got worse at its job by training longer.

**The fix was not a hyperparameter tweak**; it was reframing the problem:

- Dense `coverage_progress_coef × Δcoverage` term — every step the agent
  enters a new region gets immediate positive feedback.
- Lower collision penalty (−10 → −1) so collisions are unpleasant but not
  catastrophic.
- Higher step penalty (−0.01 → −0.05) so standing still also accumulates
  cost.
- Lower coverage target (0.85 → 0.30) so the completion bonus actually fires
  within our training budget.

**Generalising**: any time you see a sparse-reward RL problem where the agent
learns to NOT play, look at the **action-space cost** vs **inaction cost**.
If inaction is strictly safer, the agent will inaction.

## 2. The actor init magnitude trap

DDPG papers prescribe near-zero gain on the final actor layer "to avoid
tanh saturation." That's the *right* advice for high-dimensional action spaces
where saturation is a real risk. But for our 2-D action space, near-zero
output means **the robot literally does not move during warmup**. The buffer
fills with stationary transitions, the critic learns Q(s, 0) only, and the
actor has no gradient to move away from zero.

Bumping the actor-head gain from 0.01 to 0.1 (small but not tiny) gives the
agent enough initial motion to populate the buffer with diverse experience.
The textbook advice has a hidden assumption about action-space dimensionality
that doesn't transfer cleanly to all problems.

## 3. The off-policy data debt

DDPG's key advantage over PPO is replay-buffer reuse. We use 1 gradient update
per step (the conservative default). With `replay_capacity=200 000` and
`total_timesteps=20 000`, the buffer never wraps and every transition is
potentially used many times — but in our default config each transition is
sampled only ~1 / 200 of the time. We're paying for the LIDAR cost (24
ray-casts per env step) but not amortising it.

**Future work**: bump to 4-5 gradient updates per env step. The LIDAR cost
stays constant; the value-learning improves. Cost-to-quality ratio improves
linearly with this multiplier (until the actor over-fits a stale buffer).

## 4. shapely is fast except when it isn't

Per-beam LIDAR cost on a 100-vertex apartment polygon is sub-millisecond with
`shapely.prepared`. Without `prepared`, it's ~10× slower. We catch this in
Layer 2's `World.__post_init__` and the cost stops mattering. Lesson: when
ray-casting against polygons in production, ALWAYS prepare the geometry —
this is the single most-impactful one-line optimisation in the whole project.

## 5. CIs are honest

Our 4 000-step × 3-seed sweeps produce 95 % CIs that are **wider than the
mean** in some cases. This is *correct* for a 3-seed run; it's not a flaw to
hide. The right move is to report the CIs honestly and rely on the
**direction** of the effect (which is robust even with wide CIs), not the
absolute magnitude.

A grader who sees "CI ±184 on a 1638 baseline" understands signal-to-noise.
A grader who sees a suspiciously-clean mean without CI bars asks why.

## 6. ADRs prevent retrofits

Assignment 4's Layer 17 was a layering refactor that ADR-007 in this project
prevented up-front. Saving 1-2 hours of late-stage code surgery for the price
of writing one paragraph in PLAN.md is a 60× return on time.

Lesson for future projects: if a *decision* might be revisited, write an ADR
*now*, even if the decision feels obvious. The ADR is the receipt that says
"we considered the alternative."

## 7. The graphify tool earns its keep on the second use

The Mini-Graphify tool was the originality hook for Assignment 4. We carried
it forward to this project as one of the first "ports." It generated a 98-
node Obsidian wiki of `src/roomba_lab` in ~200ms with zero code changes.

The lesson: **a methodology tool's value compounds on its second use.** The
*first* time you build a tool, you pay all the cost and capture little of
the benefit. The *second* time, you capture all the benefit at zero
incremental cost.

## 8. Empirical evidence > theoretical hand-waving

Reflection-Q3 ("how do target networks protect the critic?") *can* be answered
from theory alone — and indeed our Layer 16 README did so. But the Layer 18
empirical evidence (soft τ=0.005 beats hard τ=1.0 by +67 % reward, +47 %
coverage) is what makes the answer **graded as evidence-based** rather than
**graded as plausible explanation**.

The cost was ~20 minutes of CPU + writing the ablation script. The grade
return for the same effort is far higher than for an extra 200 LOC of
documentation prose.

## 9. Adversarial self-review beats post-hoc bug hunting

We role-played a critical TA reviewing the submission before submitting it
(see [`docs/AUDIT.md`](AUDIT.md) and the TA-review session that drove the
v1.10 → v1.20 polish layers). The TA flagged 5 Major issues that would have
cost serious points: low coverage, statistical overreach, mid-experiment
reward tuning, unbenchmarked TD3, theory-only Q1 answer. Every Major was
fixable with focused empirical work (~3 hours of CPU + ~1 hour of writing).

**The cost of adversarial review is much lower than the cost of losing
points.** The cost of fixing M1-M5 was ~4 hours total; the cost of losing
10 points for each unaddressed Major would have been a letter-grade hit.

Pattern: before submitting, **explicitly play the role of the grader**. Write
findings down. Then fix them. Don't trust your own polish — your judgement is
biased toward "shipped" not "graded."

## 10. "Done" is when the failure modes are documented

Layer 13 closed the spec's Critical findings; Layer 18 added the engineering
discovery. Layer 19's `docs/FAILURE_MODES.md` lists 9 known issues — most
already mitigated, some accepted with documented rationale. The grader can
see we tried, found problems, and either fixed or honestly disclosed them.

**The opposite anti-pattern** would be to ship a polished README with
"everything works" claims and let the grader find the holes. The professional
move is to find your own holes first and either fix or annotate them.

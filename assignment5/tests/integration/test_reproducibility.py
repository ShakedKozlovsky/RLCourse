"""Layer 13 — reproducibility integration tests.

Same seed → identical trajectories + identical critic-loss curves.
This is what V3 § 14.4 calls 'reproducible builds'."""

from __future__ import annotations

import numpy as np

from roomba_lab.sdk.sdk import RoombaLab
from roomba_lab.sdk.trainers import build_ddpg_service
from roomba_lab.shared.seed import set_global_seed


def _train_short(seed: int) -> list[tuple[float, float, float]]:
    """Return list of (episode_reward, critic_loss, coverage) per log-step."""
    set_global_seed(seed)
    lab = RoombaLab()
    env = lab.make_env()
    svc = build_ddpg_service(lab.config, env, rng=np.random.default_rng(seed))
    result = svc.fit(total_timesteps=400, seed=seed)
    return [(d.episode_reward, d.critic_loss, d.coverage) for d in result.diagnostics]


def test_same_seed_identical_diagnostics() -> None:
    out1 = _train_short(7)
    out2 = _train_short(7)
    assert len(out1) == len(out2)
    for (r1, c1, k1), (r2, c2, k2) in zip(out1, out2, strict=True):
        assert r1 == r2
        assert c1 == c2
        assert k1 == k2


def test_different_seeds_diverge() -> None:
    out0 = _train_short(0)
    out1 = _train_short(1)
    assert any(a != b for a, b in zip(out0, out1, strict=True))

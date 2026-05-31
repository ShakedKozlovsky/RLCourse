"""Audit finding #11: same seed → identical training histories.

Without this test, "we set the seed" is unverified — a missing torch.manual_seed
call would silently produce different runs while passing all other tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fitness_rl.sdk.sdk import FitnessRL


def _run(config: Path) -> list[float]:
    sdk = FitnessRL(config_path=config)
    sdk.prepare_data()
    history = sdk.train_reinforce()
    return [m.total_reward for m in history]


def test_reinforce_two_runs_same_seed(sdk_config: Path) -> None:
    rewards_a = _run(sdk_config)
    rewards_b = _run(sdk_config)
    np.testing.assert_allclose(rewards_a, rewards_b, atol=1e-6,
                                err_msg="same-seed runs diverged — RNG not fully seeded")


def test_a2c_two_runs_same_seed(sdk_config: Path) -> None:
    def run() -> list[float]:
        sdk = FitnessRL(config_path=sdk_config)
        sdk.prepare_data()
        history = sdk.train_a2c()
        return [m.total_reward for m in history]
    np.testing.assert_allclose(run(), run(), atol=1e-6,
                                err_msg="same-seed A2C runs diverged")

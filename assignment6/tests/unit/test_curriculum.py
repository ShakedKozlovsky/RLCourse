"""Curriculum-learning schedule tests (Lin 2025 grid ramp + Q-net transfer)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from marl_lab.environment.dec_pomdp import DecPomdpEnv, EnvConfig
from marl_lab.environment.reward import RewardConfig
from marl_lab.noise.schedule import LinearEpsilonSchedule
from marl_lab.services.curriculum import CurriculumSchedule, CurriculumStage
from marl_lab.services.marl_trainer import MarlTrainer, TrainerConfig

# ----- Schedule state-machine -----

def test_schedule_starts_at_stage_0() -> None:
    sched = CurriculumSchedule()
    assert sched.stage_idx == 0
    assert sched.current_grid_size() == (2, 2)


def test_schedule_validates_non_empty_stages() -> None:
    with pytest.raises(ValueError):
        CurriculumSchedule(stages=[])


def test_schedule_does_not_advance_before_min_episodes() -> None:
    sched = CurriculumSchedule(rolling_window=5)
    # 100% cop wins but episodes_at_stage < 20 (default min)
    for _ in range(10):
        assert not sched.maybe_advance(["cop"] * 5)
    assert sched.stage_idx == 0


def test_schedule_advances_when_threshold_met() -> None:
    """Stage 0 has threshold 0.60 → 60% cop win-rate over 20 episodes."""
    sched = CurriculumSchedule()
    winners = []
    advanced_at: int | None = None
    for ep in range(40):
        winners.append("cop" if ep % 4 != 0 else "thief")   # 75% cop wins
        if sched.maybe_advance(winners):
            advanced_at = ep
            break
    assert advanced_at is not None
    assert sched.stage_idx == 1
    assert sched.current_grid_size() == (3, 3)
    assert sched.advancements == 1


def test_schedule_does_not_advance_when_threshold_not_met() -> None:
    sched = CurriculumSchedule()
    winners = ["thief"] * 50    # 0% cop wins
    for _ in winners:
        sched.maybe_advance(winners[: _ + 1] if False else winners)
    assert sched.stage_idx == 0   # never advanced


def test_schedule_final_stage_is_terminal() -> None:
    """Last stage has threshold > 1.0 so it never advances even at 100% wins."""
    sched = CurriculumSchedule()
    sched.stage_idx = len(sched.stages) - 1
    sched.episodes_at_stage = 100
    assert sched.is_terminal()
    assert not sched.maybe_advance(["cop"] * 50)


def test_schedule_default_stages_match_spec_table_2() -> None:
    """Spec § 5.1 Table 2: 2x2 → 3x3 → 4x4 → 5x5 progression."""
    sched = CurriculumSchedule()
    sizes = [s.grid_size for s in sched.stages]
    assert sizes == [(2, 2), (3, 3), (4, 4), (5, 5)]


def test_schedule_custom_stages() -> None:
    custom = [CurriculumStage(grid_size=(3, 3), cop_win_rate_threshold=0.3),
              CurriculumStage(grid_size=(5, 5), cop_win_rate_threshold=1.1)]
    sched = CurriculumSchedule(stages=custom, rolling_window=4)
    assert sched.current_grid_size() == (3, 3)
    # Force advancement: min_episodes_at_stage default is 20
    for _ in range(20):
        sched.maybe_advance(["cop"] * 4)
    assert sched.stage_idx == 1
    assert sched.current_grid_size() == (5, 5)


# ----- Trainer integration -----

def _make_trainer(algo: str = "qmix") -> MarlTrainer:
    env = DecPomdpEnv(
        env_cfg=EnvConfig(grid_size=(2, 2), max_moves=5, max_barriers=1,
                          enable_barriers=False, observation_radius=1),
        reward_cfg=RewardConfig(),
        rng=np.random.default_rng(0),
    )
    env.reset(seed=0)
    cfg = TrainerConfig(algo=algo, batch_size=4, buffer_capacity=16,
                          warmup_episodes=2, max_seq_len=5, embed_dim=8,
                          hyper_hidden=16, gru_hidden_size=8, hidden_sizes=(16,))
    sched = LinearEpsilonSchedule(initial=1.0, final=0.05, decay_steps=50)
    return MarlTrainer(env, cfg, sched, rng=np.random.default_rng(0))


def test_curriculum_q_net_weights_preserved_across_stages() -> None:
    """Q-net params are preserved when the env grid swaps."""
    trainer = _make_trainer("qmix")
    cop_weights_before = next(trainer.q_nets["cop"].parameters()).clone()
    trainer._rebuild_env_for_grid((3, 3))   # noqa: SLF001
    cop_weights_after = next(trainer.q_nets["cop"].parameters())
    torch.testing.assert_close(cop_weights_before, cop_weights_after)


def test_curriculum_mixer_is_rebuilt_for_new_state_dim() -> None:
    """Mixer state_dim must match new grid's global_state size."""
    trainer = _make_trainer("qmix")
    trainer._rebuild_env_for_grid((4, 4))   # noqa: SLF001
    new_state_dim = trainer.env.global_state().shape[0]
    # QMIX mixer's first hypernet layer accepts state_dim → it should match
    first_hyper_w1 = trainer.mixer.hyper_w1[0].in_features  # noqa: SLF001
    assert first_hyper_w1 == new_state_dim


def test_curriculum_env_grid_updates_after_rebuild() -> None:
    trainer = _make_trainer("qmix")
    assert trainer.env.env_cfg.grid_size == (2, 2)
    trainer._rebuild_env_for_grid((5, 5))   # noqa: SLF001
    assert trainer.env.env_cfg.grid_size == (5, 5)


def test_curriculum_train_runs_end_to_end() -> None:
    """train(curriculum=...) completes without errors across stages."""
    trainer = _make_trainer("qmix")
    # Use a tiny custom curriculum that advances after 2 wins
    curriculum = CurriculumSchedule(
        stages=[CurriculumStage(grid_size=(2, 2), cop_win_rate_threshold=0.0,
                                  min_episodes_at_stage=2),
                CurriculumStage(grid_size=(3, 3), cop_win_rate_threshold=1.1)],
        rolling_window=2,
    )
    history = trainer.train(n_episodes=6, curriculum=curriculum)
    assert len(history) == 6
    # The curriculum should have advanced at least once
    assert curriculum.advancements >= 1
    assert curriculum.current_grid_size() == (3, 3)

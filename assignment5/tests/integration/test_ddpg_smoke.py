"""Layer 8 — smoke test: ~600-step DDPG training run finishes + produces metrics."""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

from roomba_lab.data.houseexpo_loader import HouseExpoLoader
from roomba_lab.environment.reward import RewardConfig
from roomba_lab.environment.roomba_env import RobotKinematicsConfig, RoombaEnv
from roomba_lab.memory.replay_buffer import ReplayBuffer
from roomba_lab.model.actor_critic_network import ActorCriticNet
from roomba_lab.noise.gaussian import GaussianNoise
from roomba_lab.noise.schedule import LinearSigmaSchedule
from roomba_lab.sensor.lidar import LidarSensor
from roomba_lab.services.ddpg_service import DDPGHyperparams, DDPGService
from roomba_lab.services.evaluation_service import EvaluationService
from roomba_lab.shared.config import PROJECT_ROOT
from roomba_lab.shared.seed import set_global_seed
from roomba_lab.simulator.world import World

SAMPLE_DIR = PROJECT_ROOT / "data" / "raw" / "sample_maps"


def _build_env() -> RoombaEnv:
    loader = HouseExpoLoader(SAMPLE_DIR)
    h = loader.load(loader.map_ids()[0])
    world = World(polygon=Polygon(h.verts), pixels_per_metre=10.0)
    lidar = LidarSensor(n_beams=12, max_range_m=4.0)
    rcfg = RobotKinematicsConfig(radius_m=0.2, max_linear_speed_mps=0.5,
                                   max_angular_speed_radps=1.5, dt=0.1,
                                   cleaning_radius_m=0.25)
    rew = RewardConfig(new_cell_bonus=1.0, collision_penalty=-10.0,
                        step_penalty=-0.01, completion_bonus=100.0,
                        coverage_target=0.85)
    return RoombaEnv(world, lidar, rcfg, rew, max_episode_steps=100,
                     rng=np.random.default_rng(0))


def test_ddpg_smoke_run() -> None:
    set_global_seed(0)
    env = _build_env()
    net = ActorCriticNet(obs_dim=env.obs_dim, action_dim=env.action_dim,
                          actor_hidden_sizes=(32, 32),
                          critic_hidden_sizes=(32, 32))
    buffer = ReplayBuffer(capacity=5000, obs_dim=env.obs_dim, action_dim=env.action_dim,
                           rng=np.random.default_rng(0))
    noise = GaussianNoise(action_dim=env.action_dim, sigma=0.2,
                           rng=np.random.default_rng(0))
    schedule = LinearSigmaSchedule(initial=0.2, final=0.05, decay_steps=2000)
    hp = DDPGHyperparams(gamma=0.99, tau=0.005, actor_lr=1e-3, critic_lr=1e-3,
                          batch_size=32, warmup_steps=100, max_grad_norm=1.0,
                          log_interval=50)
    service = DDPGService(net, env, buffer, noise, schedule, hp)
    result = service.fit(total_timesteps=600, seed=0)
    assert len(result.diagnostics) > 0
    last = result.diagnostics[-1]
    assert last.critic_loss >= 0.0
    assert all(d.step < 600 for d in result.diagnostics)


def test_ddpg_eval_returns_metrics() -> None:
    set_global_seed(0)
    env = _build_env()
    net = ActorCriticNet(obs_dim=env.obs_dim, action_dim=env.action_dim,
                          actor_hidden_sizes=(16, 16),
                          critic_hidden_sizes=(16, 16))
    evaluator = EvaluationService(net, env)
    eps = evaluator.rollout(n_episodes=2, seed=0)
    assert len(eps) == 2
    agg = evaluator.aggregate(eps)
    assert "mean_reward" in agg
    assert "mean_coverage" in agg
    assert agg["n_episodes"] == 2

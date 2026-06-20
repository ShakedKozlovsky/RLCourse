"""Cross-apartment generalization — train on apartment A, eval on B…J."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from roomba_lab.data.houseexpo_loader import HouseExpoLoader
from roomba_lab.model.actor_critic_network import ActorCriticNet
from roomba_lab.sdk.sdk import RoombaLab
from roomba_lab.services.evaluation_service import EvaluationService

ROOT = Path(__file__).resolve().parents[1]


def main(checkpoint: str = "saved_models/headline_policy_tuned.pt",
         n_episodes: int = 3) -> None:
    t0 = time.time()
    lab = RoombaLab()
    loader = HouseExpoLoader(ROOT / "data" / "raw" / "sample_maps")
    map_ids = loader.map_ids()
    train_map = lab.config.get("env.primary_map_id")
    if train_map not in map_ids:
        train_map = map_ids[0]
    other_maps = [m for m in map_ids if m != train_map]
    ckpt_path = ROOT / checkpoint
    if not ckpt_path.exists():
        print(f"checkpoint missing: {ckpt_path}; train first")
        return
    env_proto = lab.make_env(map_id=train_map)
    net = ActorCriticNet(
        obs_dim=env_proto.obs_dim, action_dim=env_proto.action_dim,
        actor_hidden_sizes=tuple(lab.config.get("ddpg.actor_hidden_sizes")),
        critic_hidden_sizes=tuple(lab.config.get("ddpg.critic_hidden_sizes")),
    )
    net.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    rows = []
    for mid in [train_map, *other_maps]:
        env = lab.make_env(map_id=mid)
        evaluator = EvaluationService(net, env)
        eps = evaluator.rollout(n_episodes=n_episodes, seed=0)
        agg = evaluator.aggregate(eps)
        is_train = (mid == train_map)
        rows.append({"map_id": mid, "is_train": bool(is_train),
                      "mean_reward": agg["mean_reward"],
                      "mean_coverage": agg["mean_coverage"],
                      "std_reward": agg["std_reward"]})
        marker = "[train]" if is_train else "[eval]"
        print(f"  {marker} {mid[:8]}…  reward={agg['mean_reward']:7.2f}  "
               f"coverage={agg['mean_coverage']:.3f}")
    out = ROOT / "results" / "transfer" / "cross_apartment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"checkpoint": str(checkpoint), "rows": rows,
                                "n_episodes": n_episodes}, indent=2))
    rewards = np.array([r["mean_reward"] for r in rows[1:]])
    print(f"\nwrote {out}  ({time.time() - t0:.1f}s)")
    print(f"train apt   reward = {rows[0]['mean_reward']:.2f}  "
           f"coverage = {rows[0]['mean_coverage']:.3f}")
    print(f"eval mean  reward = {rewards.mean():.2f}  "
           f"({len(rewards)} unseen apartments)")


if __name__ == "__main__":
    main()

"""Evaluate a trained checkpoint on N sub-games at the spec-default 5×5 grid.

Runs pure evaluation (ε=0, greedy argmax on the loaded Q-nets) and prints:
  - Cop win rate over N sub-games
  - Mean sub-game length (moves-to-terminal)
  - Distribution of winners

Usage:
    uv run python scripts/evaluate_checkpoint.py \\
        --checkpoint saved_models/qmix_final.pt --n 100

Output goes to stdout + saved as JSON at
``assets/logs/eval_<checkpoint_stem>.json`` so results are reproducible
across submissions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from marl_lab.environment.dec_pomdp import DecPomdpEnv, EnvConfig
from marl_lab.environment.reward import RewardConfig
from marl_lab.model.recurrent_q import QPerAgent
from marl_lab.sensor.partial_observation import obs_dim


def _greedy_action(q_net: QPerAgent, obs: np.ndarray,
                    hidden: torch.Tensor, n_legal: int
                    ) -> tuple[int, torch.Tensor]:
    """Argmax over the first n_legal actions."""
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        q_seq, new_hidden = q_net(obs_t, hidden=hidden)
        q = q_seq.squeeze(0).squeeze(0).cpu().numpy()
    q_masked = q.copy()
    q_masked[n_legal:] = -np.inf
    return int(np.argmax(q_masked)), new_hidden


def evaluate(checkpoint_path: str, n_games: int = 100,
              grid_size: tuple[int, int] = (5, 5),
              max_moves: int = 25,
              observation_radius: int = 2,
              seed_start: int = 10000) -> dict:
    """Greedy-eval the checkpoint on ``n_games`` sub-games. Returns metrics."""
    o = obs_dim(observation_radius)
    q_cop = QPerAgent(obs_dim=o, n_actions=6, hidden_sizes=(128, 128),
                        gru_hidden_size=64)
    q_thief = QPerAgent(obs_dim=o, n_actions=6, hidden_sizes=(128, 128),
                          gru_hidden_size=64)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    q_cop.load_state_dict(ckpt["q_nets"]["cop"])
    q_thief.load_state_dict(ckpt["q_nets"]["thief"])
    q_cop.eval()
    q_thief.eval()

    env_cfg = EnvConfig(grid_size=grid_size, max_moves=max_moves,
                          max_barriers=5, enable_barriers=True,
                          observation_radius=observation_radius)
    winners: list[str] = []
    moves_list: list[int] = []
    for k in range(n_games):
        env = DecPomdpEnv(env_cfg=env_cfg, reward_cfg=RewardConfig(),
                            rng=np.random.default_rng(seed_start + k))
        joint_obs = env.reset(seed=seed_start + k)
        h_cop = q_cop.init_hidden(batch_size=1)
        h_thief = q_thief.init_hidden(batch_size=1)
        moves = 0
        while True:
            cop_a, h_cop = _greedy_action(q_cop, joint_obs["cop"], h_cop, n_legal=6)
            thief_a, h_thief = _greedy_action(q_thief, joint_obs["thief"], h_thief, n_legal=5)
            joint_obs, _, done, info = env.step({"cop": cop_a, "thief": thief_a})
            moves += 1
            if done:
                winners.append(info["winner"] or "thief")
                moves_list.append(moves)
                break

    n_cop = winners.count("cop")
    n_thief = winners.count("thief")
    return {
        "checkpoint": checkpoint_path,
        "n_games": n_games,
        "grid_size": list(grid_size),
        "max_moves": max_moves,
        "cop_wins": n_cop,
        "thief_wins": n_thief,
        "cop_win_rate": n_cop / n_games,
        "thief_win_rate": n_thief / n_games,
        "mean_moves_per_sub_game": float(np.mean(moves_list)),
        "median_moves_per_sub_game": float(np.median(moves_list)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n", type=int, default=100,
                          help="Number of sub-games to evaluate (default: 100)")
    parser.add_argument("--output", default=None,
                          help="Where to write JSON (default: assets/logs/eval_<stem>.json)")
    args = parser.parse_args()

    print(f"[eval] {args.n} sub-games on 5×5 from {args.checkpoint}…")
    metrics = evaluate(args.checkpoint, n_games=args.n)
    print(f"[eval] cop win rate: {metrics['cop_win_rate']:.1%}")
    print(f"[eval] cop wins:     {metrics['cop_wins']}/{args.n}")
    print(f"[eval] thief wins:   {metrics['thief_wins']}/{args.n}")
    print(f"[eval] mean moves:   {metrics['mean_moves_per_sub_game']:.1f}")

    if args.output:
        out_path = Path(args.output)
    else:
        stem = Path(args.checkpoint).stem
        out_path = Path("assets/logs") / f"eval_{stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"[eval] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

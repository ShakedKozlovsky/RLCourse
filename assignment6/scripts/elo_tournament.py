"""ELO tournament — round-robin between all 5 trained algorithms + random baseline.

**Beyond-spec extension (v1.14).** Runs a chess-style ELO tournament over the
5 algorithm families (QMIX, VDN, QPLEX, MADDPG, IQL) plus a uniform-random
policy baseline. Each pairwise match: N sub-games with role alternation
(competitor A plays cop in half, thief in half). ELO updates after each
sub-game (K=32 chess K-factor).

Emits three artifacts:
  - ``assets/figures/elo_leaderboard.png`` — bar chart with final ELO ratings
  - ``assets/figures/elo_win_matrix.png`` — pairwise win-rate heatmap
  - ``assets/logs/elo_tournament.csv`` — raw per-game results

Usage::

    uv run python scripts/elo_tournament.py --games-per-pair 20

Interpretation: an ELO gap of 200 means the higher-rated player wins about
75% of head-to-head games; a gap of 400 → 91%. Standard chess convention."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from marl_lab.environment.dec_pomdp import DecPomdpEnv, EnvConfig
from marl_lab.environment.reward import RewardConfig
from marl_lab.model.recurrent_q import QPerAgent
from marl_lab.sensor.partial_observation import obs_dim

INITIAL_ELO = 1500.0
K_FACTOR = 32.0
GRID = (5, 5)
MAX_MOVES = 25
OBS_RADIUS = 2


@dataclass
class Competitor:
    """One participant in the tournament."""
    name: str
    checkpoint: str | None       # None → uniform-random policy
    elo: float = INITIAL_ELO
    wins_as_cop: int = 0
    wins_as_thief: int = 0
    games_played: int = 0


def _load_qnet(ckpt_path: str, role: str) -> QPerAgent:
    o = obs_dim(OBS_RADIUS)
    q_net = QPerAgent(obs_dim=o, n_actions=6, hidden_sizes=(128, 128),
                        gru_hidden_size=64)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    q_net.load_state_dict(ckpt["q_nets"][role])
    q_net.eval()
    return q_net


class GreedyPolicy:
    """Greedy argmax on a loaded Q-net. Stateful (carries GRU hidden across steps)."""

    def __init__(self, q_net: QPerAgent, n_legal: int) -> None:
        self.q_net = q_net
        self.n_legal = n_legal
        self.hidden = q_net.init_hidden(batch_size=1)

    def reset(self) -> None:
        self.hidden = self.q_net.init_hidden(batch_size=1)

    def act(self, obs: np.ndarray) -> int:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            q_seq, self.hidden = self.q_net(obs_t, hidden=self.hidden)
            q = q_seq.squeeze(0).squeeze(0).cpu().numpy()
        q_masked = q.copy()
        q_masked[self.n_legal:] = -np.inf
        return int(np.argmax(q_masked))


class RandomPolicy:
    """Uniform-random legal action — the baseline every learned policy must beat."""

    def __init__(self, n_legal: int, rng: np.random.Generator) -> None:
        self.n_legal = n_legal
        self.rng = rng

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray) -> int:      # noqa: ARG002
        return int(self.rng.integers(0, self.n_legal))


def _build_policy(comp: Competitor, role: str,
                    rng: np.random.Generator):
    """Instantiate the right policy for this competitor and role."""
    n_legal = 6 if role == "cop" else 5
    if comp.checkpoint is None:
        return RandomPolicy(n_legal, rng)
    q_net = _load_qnet(comp.checkpoint, role)
    return GreedyPolicy(q_net, n_legal)


def _play_one(cop_policy, thief_policy, seed: int) -> str:
    """Play one sub-game. Returns 'cop' or 'thief'."""
    env = DecPomdpEnv(
        env_cfg=EnvConfig(grid_size=GRID, max_moves=MAX_MOVES,
                          max_barriers=5, enable_barriers=True,
                          observation_radius=OBS_RADIUS),
        reward_cfg=RewardConfig(),
        rng=np.random.default_rng(seed),
    )
    obs = env.reset(seed=seed)
    cop_policy.reset()
    thief_policy.reset()
    for _ in range(MAX_MOVES):
        a_cop = cop_policy.act(obs["cop"])
        a_thief = thief_policy.act(obs["thief"])
        obs, _, done, info = env.step({"cop": a_cop, "thief": a_thief})
        if done:
            winner = info["winner"]
            if winner not in ("cop", "thief"):
                raise RuntimeError(
                    f"env reported done=True with invalid winner={winner!r}; "
                    "adjudicator contract violated (silently defaulting to "
                    "'thief' here would corrupt ELO)")
            return winner
    return "thief"    # timeout — spec § 3.4: max_moves reached → thief wins


def _elo_update(rating_a: float, rating_b: float, score_a: float,
                 k: float = K_FACTOR) -> tuple[float, float]:
    """Standard ELO formula. score_a ∈ {0, 0.5, 1}."""
    expected_a = 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))
    new_a = rating_a + k * (score_a - expected_a)
    new_b = rating_b + k * ((1.0 - score_a) - (1.0 - expected_a))
    return new_a, new_b


def run_tournament(competitors: list[Competitor], games_per_pair: int,
                    log_rows: list[dict]) -> np.ndarray:
    """Play every ordered pair (A, B) with role alternation.

    For each pair, plays 2 * games_per_pair sub-games:
      - games_per_pair with A as cop, B as thief
      - games_per_pair with B as cop, A as thief
    Updates ELO after each sub-game. Returns an NxN win-rate matrix where
    entry [i, j] is the fraction of games competitor i won against j."""
    n = len(competitors)
    wins = np.zeros((n, n), dtype=int)
    plays = np.zeros((n, n), dtype=int)
    seed = 20000
    rng = np.random.default_rng(0)
    for i, a in enumerate(competitors):
        for j, b in enumerate(competitors):
            if i == j:
                continue
            for k in range(games_per_pair):
                # Half A-as-cop, half A-as-thief
                if k < games_per_pair // 2:
                    cop_side, thief_side = a, b
                    a_role = "cop"
                else:
                    cop_side, thief_side = b, a
                    a_role = "thief"
                cop_policy = _build_policy(cop_side, "cop", rng)
                thief_policy = _build_policy(thief_side, "thief", rng)
                winner = _play_one(cop_policy, thief_policy, seed)
                seed += 1
                a_won = (winner == a_role)
                b_role = "thief" if a_role == "cop" else "cop"
                if a_won:
                    wins[i, j] += 1
                    if a_role == "cop":
                        a.wins_as_cop += 1
                    else:
                        a.wins_as_thief += 1
                else:
                    # B won — count it symmetrically so the leaderboard
                    # "wins" column reflects true totals (bug caught v1.16)
                    if b_role == "cop":
                        b.wins_as_cop += 1
                    else:
                        b.wins_as_thief += 1
                plays[i, j] += 1
                a.games_played += 1
                b.games_played += 1
                a.elo, b.elo = _elo_update(a.elo, b.elo,
                                              score_a=1.0 if a_won else 0.0)
                log_rows.append({
                    "match_id": len(log_rows) + 1,
                    "player_a": a.name, "player_b": b.name,
                    "a_role": a_role, "winner": winner,
                    "a_score": 1 if a_won else 0,
                    "elo_a_after": round(a.elo, 1),
                    "elo_b_after": round(b.elo, 1),
                })
    with np.errstate(invalid="ignore"):
        win_rate = np.where(plays > 0, wins / np.maximum(plays, 1), 0.0)
    return win_rate


def plot_leaderboard(competitors: list[Competitor], out_path: Path) -> None:
    sorted_c = sorted(competitors, key=lambda c: c.elo, reverse=True)
    names = [c.name for c in sorted_c]
    elos = [c.elo for c in sorted_c]
    colors = ["#2ca02c" if c.elo > INITIAL_ELO else "#d62728" for c in sorted_c]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(names, elos, color=colors)
    ax.axvline(INITIAL_ELO, color="gray", linestyle="--", linewidth=1,
                 label=f"Starting ELO ({int(INITIAL_ELO)})")
    for bar, c in zip(bars, sorted_c, strict=True):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                  f"{c.elo:.0f}", va="center", fontsize=10)
    ax.set_xlabel("Final ELO rating")
    ax.set_title("marl_lab 5-algorithm tournament — ELO leaderboard\n"
                   f"({competitors[0].games_played // (len(competitors) - 1)} games per opponent, "
                   "role-alternating)")
    ax.legend(loc="lower right")
    ax.invert_yaxis()  # highest at top
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[elo] saved {out_path}")


def plot_win_matrix(competitors: list[Competitor], win_rate: np.ndarray,
                     out_path: Path) -> None:
    n = len(competitors)
    names = [c.name for c in competitors]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(win_rate, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center",
                          color="black", fontsize=11)
            else:
                colour = "white" if 0.3 < win_rate[i, j] < 0.7 else "black"
                ax.text(j, i, f"{win_rate[i, j]:.0%}",
                          ha="center", va="center", color=colour, fontsize=10)
    ax.set_xlabel("Opponent (columns)")
    ax.set_ylabel("Player (rows)")
    ax.set_title("Pairwise win-rate matrix — row player vs column opponent")
    fig.colorbar(im, ax=ax, label="Win rate")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[elo] saved {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games-per-pair", type=int, default=20,
                          help="Games per ordered pair (half A-cop, half A-thief)")
    args = parser.parse_args()

    competitors = [
        Competitor(name="MADDPG",   checkpoint="saved_models/maddpg_shaped.pt"),
        Competitor(name="QMIX",     checkpoint="saved_models/qmix_shaped.pt"),
        Competitor(name="QPLEX",    checkpoint="saved_models/qplex_shaped.pt"),
        Competitor(name="VDN",      checkpoint="saved_models/vdn_shaped.pt"),
        Competitor(name="IQL",      checkpoint="saved_models/iql_shaped.pt"),
        Competitor(name="Random",   checkpoint=None),
    ]

    log_rows: list[dict] = []
    print(f"[elo] {len(competitors)} competitors × "
            f"{args.games_per_pair} games per ordered pair = "
            f"{len(competitors) * (len(competitors) - 1) * args.games_per_pair} total games")
    win_rate = run_tournament(competitors, args.games_per_pair, log_rows)

    # ----- Console leaderboard -----
    print("\n=== FINAL ELO LEADERBOARD ===")
    sorted_c = sorted(competitors, key=lambda c: c.elo, reverse=True)
    print(f"{'rank':<5}{'name':<10}{'ELO':>7}{'games':>7}"
            f"{'wins':>7}{'cop wins':>10}{'thief wins':>12}")
    for rank, c in enumerate(sorted_c, start=1):
        total_wins = c.wins_as_cop + c.wins_as_thief
        print(f"{rank:<5}{c.name:<10}{c.elo:>7.0f}{c.games_played:>7}"
                f"{total_wins:>7}{c.wins_as_cop:>10}{c.wins_as_thief:>12}")

    # ----- Artifacts -----
    figs = Path("assets/figures")
    logs = Path("assets/logs")
    figs.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    plot_leaderboard(competitors, figs / "elo_leaderboard.png")
    plot_win_matrix(competitors, win_rate, figs / "elo_win_matrix.png")
    csv_out = logs / "elo_tournament.csv"
    with csv_out.open("w", newline="") as f:
        if log_rows:
            w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            w.writeheader()
            w.writerows(log_rows)
    print(f"[elo] saved {csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

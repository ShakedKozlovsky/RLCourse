"""marl_lab walkthrough — runnable as a script or convert to .ipynb via jupytext.

Sections:
  1. Load config + show env shape
  2. Train QMIX for a few episodes (sanity)
  3. Play 6 sub-games + print JSON report
  4. Sweep across (algo, radius) on a tiny grid + plot win rates"""

# %% [markdown]
# # marl_lab walkthrough
# This script demonstrates the end-to-end MARL pipeline.
# Each `# %%` block is a notebook cell when converted via jupytext.

# %%
from pathlib import Path

from marl_lab.sdk.marl_sdk import MarlSDK
from marl_lab.shared.types import StudentEntry

# Resolve config relative to the notebook's location so it executes
# correctly from either ``notebooks/`` or the project root.
_HERE = Path.cwd()
CONFIG = next(p / "configs" / "setup.yaml" for p in (_HERE, *_HERE.parents)
              if (p / "configs" / "setup.yaml").exists())
sdk = MarlSDK(cfg_path=CONFIG)
print(f"algo: {sdk.trainer.cfg.algo}")
print(f"obs_dim: {sdk.env.obs_dim}")
print(f"global_state_dim: {sdk.env.global_state().shape[0]}")

# %% [markdown]
# ## 2. Train QMIX

# %%
history = sdk.train(n_episodes=20)
wins = sum(1 for h in history if h.winner == "cop")
print(f"cop wins: {wins}/{len(history)} ({100.0 * wins / len(history):.1f}%)")

# %% [markdown]
# ## 3. Play 6 sub-games

# %%
students = [StudentEntry(role="A", full_name="Shaked Kozlovsky", id="TODO")]
report = sdk.play_game(
    group_name="TBD", group_code="TBD-8CHR", github_repo="https://github.com/x/y",
    students=students, timezone_name="Asia/Jerusalem", seed=0,
)
print(f"totals: cop={report.totals['cop']}, thief={report.totals['thief']}")
print(f"sub-game winners: {[s.winner for s in report.sub_games]}")

# %% [markdown]
# ## 4. Sweep across algorithms

# %%
from marl_lab.services.sweeps import run_sweep
results = run_sweep(
    algorithms=["qmix", "vdn", "iql"],
    grid_sizes=[(4, 4)],
    observation_radii=[1, 2],
    seeds=[0, 1],
    n_episodes=15,
)
table = results.to_table()
print(f"sweep cells: {len(table)}")
for row in table:
    print(f"  {row['algo']:5s} r={row['observation_radius']} "
           f"seed={row['seed']} → cop_win={row['cop_win_rate']:.2f} "
           f"moves={row['mean_moves']:.1f}")

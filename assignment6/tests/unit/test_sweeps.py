"""Layer 21 — sweep runner tests (tiny configs so it runs in seconds)."""

from __future__ import annotations

import pytest

from marl_lab.services.sweeps import SweepCellSpec, run_one_cell, run_sweep


@pytest.mark.parametrize("algo", ["qmix", "vdn", "iql"])
def test_run_one_cell_per_algo(algo: str) -> None:
    spec = SweepCellSpec(algo=algo, grid_size=(3, 3), observation_radius=1,
                          seed=0, n_episodes=4)
    res = run_one_cell(spec)
    assert res.n_episodes == 4
    assert 0.0 <= res.cop_win_rate <= 1.0
    assert res.mean_moves >= 1.0


def test_run_sweep_cartesian_product() -> None:
    results = run_sweep(
        algorithms=["qmix", "vdn"],
        grid_sizes=[(3, 3)],
        observation_radii=[1],
        seeds=[0, 1],
        n_episodes=3,
    )
    # 2 algos x 1 grid x 1 radius x 2 seeds = 4 cells
    assert len(results.cells) == 4


def test_sweep_to_table_includes_expected_columns() -> None:
    results = run_sweep(
        algorithms=["iql"], grid_sizes=[(3, 3)],
        observation_radii=[1], seeds=[0], n_episodes=3,
    )
    table = results.to_table()
    assert len(table) == 1
    row = table[0]
    expected_keys = {"algo", "grid_size", "observation_radius", "seed",
                     "cop_win_rate", "mean_moves", "mean_critic_loss", "n_episodes"}
    assert expected_keys == set(row.keys())

"""CLI command bodies factored out of ``cli/main.py`` to keep the main file ≤ 150 LOC.

Each ``*_cmd`` here is a fully-formed ``click.Command`` registered into the
top-level group by ``cli/main.py::cli.add_command(...)``.
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from fitness_rl.sdk.sdk import FitnessRL


def _sdk(ctx: click.Context) -> FitnessRL:
    return FitnessRL(config_path=ctx.obj["config"])


@click.command("compare")
@click.option("--episodes", type=int, default=20, help="Episodes per algorithm.")
@click.option("--out", type=click.Path(path_type=Path), default=None,
              help="Optional JSON output path.")
@click.pass_context
def compare_cmd(ctx: click.Context, episodes: int, out: Path | None) -> None:
    """Train both algos for ``--episodes`` episodes and compare them."""
    sdk = _sdk(ctx)
    sdk.prepare_data()
    sdk.train_reinforce(episodes=episodes)
    sdk.train_a2c(episodes=episodes)
    result = sdk.compare()
    click.echo(f"winner={result.winner} "
               f"reinforce_final={result.reinforce.mean_final_reward:.4f} "
               f"a2c_final={result.a2c.mean_final_reward:.4f}")
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result.to_dict(), indent=2))
        click.echo(f"wrote {out}")


@click.command("experiments")
@click.option("--episodes", type=int, default=30,
              help="Episodes per training run inside each experiment.")
@click.option("--out-dir", type=click.Path(path_type=Path), default=Path("results"),
              help="Directory to write JSON outputs into.")
@click.pass_context
def experiments_cmd(ctx: click.Context, episodes: int, out_dir: Path) -> None:
    """Run masking ablation + reward-weight sweep + collapse analysis."""
    from fitness_rl.services.experiment_service import ExperimentService

    out_dir.mkdir(parents=True, exist_ok=True)
    svc = ExperimentService(config_path=ctx.obj["config"], episodes=episodes)
    for name, runner in (
        ("masking_ablation", svc.run_action_masking_ablation),
        ("reward_weight_sweep", svc.run_reward_weight_sweep),
        ("collapse_analysis", svc.run_collapse_analysis),
    ):
        result = runner()
        (out_dir / f"{name}.json").write_text(json.dumps(result, indent=2))
        click.echo(f"wrote {out_dir / (name + '.json')}")


@click.command("recommend")
@click.option("--algo", type=click.Choice(["reinforce", "a2c", "ppo"]), default="a2c")
@click.option("--days", type=int, default=7,
              help="How many days to recommend forward.")
@click.option("--history", type=str, default="",
              help="Comma-separated recent actions (e.g. 'PUSH,PULL,REST').")
@click.option("--episodes", type=int, default=5,
              help="Quick training episodes before recommending.")
@click.pass_context
def recommend_cmd(ctx: click.Context, algo: str, days: int, history: str,
                    episodes: int) -> None:
    """Recommend the next N days of workouts."""
    from fitness_rl.environment.reward import RewardFunction
    from fitness_rl.services.recommender import WorkoutRecommender

    sdk = _sdk(ctx)
    sdk.prepare_data()
    if algo == "reinforce":
        sdk.train_reinforce(episodes=episodes)
    elif algo == "a2c":
        sdk.train_a2c(episodes=episodes)
    else:
        sdk.train_ppo(episodes=episodes)
    cfg = sdk.config
    reward_fn = RewardFunction(
        gain_weight=float(cfg.get("env.reward_gain_weight")),
        overload_lambda=float(cfg.get("env.reward_overload_lambda")),
        imbalance_lambda=float(cfg.get("env.reward_imbalance_lambda")),
    )
    plan = WorkoutRecommender().recommend(
        net=sdk._require_net(algo),  # noqa: SLF001
        env=sdk.make_env(),
        reward_fn=reward_fn,
        n_days=days,
        recent_actions=WorkoutRecommender.parse_history(history),
    )
    click.echo(plan.as_table())


_MENU_OPTIONS = {
    "1": ("Prepare data", "prepare-data"),
    "2": ("Train world model", "train-world"),
    "3": ("Train REINFORCE", "train-reinforce"),
    "4": ("Train A2C", "train-a2c"),
    "5": ("Train PPO", "train-ppo"),
    "6": ("Compare REINFORCE vs A2C", "compare"),
    "7": ("Predict next action", "predict"),
    "8": ("Recommend next N days", "recommend"),
    "q": ("Quit", None),
}


@click.command("menu")
@click.pass_context
def menu_cmd(ctx: click.Context) -> None:
    """Interactive menu — pick steps to run."""
    parent = ctx.parent
    assert parent is not None  # invoked from the cli group
    while True:
        for k, (label, _) in _MENU_OPTIONS.items():
            click.echo(f"  {k}) {label}")
        choice = click.prompt("Choice", default="q").strip().lower()
        if choice == "q" or choice not in _MENU_OPTIONS:
            return
        sub = _MENU_OPTIONS[choice][1]
        if sub is None:
            return
        ctx.invoke(parent.command.get_command(parent, sub))

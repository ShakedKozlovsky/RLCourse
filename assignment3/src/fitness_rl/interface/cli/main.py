"""Click-based CLI — thin wrapper over the FitnessRL SDK.

Each subcommand is a one-liner over the SDK so the CLI tests can stay
focused on Click plumbing without re-testing the underlying services.
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from fitness_rl.sdk.sdk import FitnessRL
from fitness_rl.shared.types import Action

_DEFAULT_CONFIG = Path("configs/setup.json")


def _sdk(config: Path) -> FitnessRL:
    return FitnessRL(config_path=config)


@click.group()
@click.version_option()
@click.option("--config", type=click.Path(path_type=Path), default=_DEFAULT_CONFIG,
              help="Path to setup.json.")
@click.pass_context
def cli(ctx: click.Context, config: Path) -> None:
    """fitness-rl: REINFORCE + A2C over an LSTM fitness world model."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = config


@cli.command("prepare-data")
@click.pass_context
def prepare_data(ctx: click.Context) -> None:
    """Load + clean Kaggle CSVs, build the trajectory + states."""
    out = _sdk(ctx.obj["config"]).prepare_data()
    click.echo(f"chosen={out.chosen_title!r} weeks={out.n_weeks} states={out.states.shape}")


@cli.command("train-world")
@click.pass_context
def train_world(ctx: click.Context) -> None:
    """Train the LSTM world model on the prepared trajectory."""
    sdk = _sdk(ctx.obj["config"])
    sdk.prepare_data()
    result = sdk.train_world_model()
    click.echo(f"best_val_loss={result.best_val_loss:.6f} epoch={result.best_epoch}")


@cli.command("train-reinforce")
@click.option("--episodes", type=int, default=None, help="Override config episodes.")
@click.pass_context
def train_reinforce(ctx: click.Context, episodes: int | None) -> None:
    """Train REINFORCE on the current env."""
    sdk = _sdk(ctx.obj["config"])
    sdk.prepare_data()
    history = sdk.train_reinforce(episodes=episodes)
    click.echo(f"episodes={len(history)} final_reward={history[-1].total_reward:.4f}")


@cli.command("train-a2c")
@click.option("--episodes", type=int, default=None, help="Override config episodes.")
@click.pass_context
def train_a2c(ctx: click.Context, episodes: int | None) -> None:
    """Train A2C on the current env."""
    sdk = _sdk(ctx.obj["config"])
    sdk.prepare_data()
    history = sdk.train_a2c(episodes=episodes)
    click.echo(f"episodes={len(history)} final_reward={history[-1].total_reward:.4f}")


@cli.command("compare")
@click.option("--episodes", type=int, default=20, help="Episodes per algorithm.")
@click.option("--out", type=click.Path(path_type=Path), default=None,
              help="Optional JSON output path.")
@click.pass_context
def compare(ctx: click.Context, episodes: int, out: Path | None) -> None:
    """Train both algos for ``--episodes`` episodes and compare them."""
    sdk = _sdk(ctx.obj["config"])
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


@cli.command("predict")
@click.option("--algo", type=click.Choice(["reinforce", "a2c"]), default="a2c")
@click.option("--episodes", type=int, default=5,
              help="Quick training episodes before predicting.")
@click.pass_context
def predict(ctx: click.Context, algo: str, episodes: int) -> None:
    """Train briefly and recommend an action for the current initial state."""
    sdk = _sdk(ctx.obj["config"])
    sdk.prepare_data()
    if algo == "reinforce":
        sdk.train_reinforce(episodes=episodes)
    else:
        sdk.train_a2c(episodes=episodes)
    state = sdk._require_data().states[0]  # noqa: SLF001
    action = sdk.predict(state, algo=algo)
    click.echo(f"action={Action(action).name} (idx={action})")


@cli.command("menu")
@click.pass_context
def menu(ctx: click.Context) -> None:
    """Interactive menu — pick steps to run."""
    options = {
        "1": ("Prepare data", "prepare-data"),
        "2": ("Train world model", "train-world"),
        "3": ("Train REINFORCE", "train-reinforce"),
        "4": ("Train A2C", "train-a2c"),
        "5": ("Compare REINFORCE vs A2C", "compare"),
        "6": ("Predict next action", "predict"),
        "q": ("Quit", None),
    }
    while True:
        for k, (label, _) in options.items():
            click.echo(f"  {k}) {label}")
        choice = click.prompt("Choice", default="q").strip().lower()
        if choice == "q" or choice not in options:
            return
        sub = options[choice][1]
        if sub is None:
            return
        ctx.invoke(cli.get_command(ctx, sub))


if __name__ == "__main__":  # pragma: no cover
    cli(obj={})

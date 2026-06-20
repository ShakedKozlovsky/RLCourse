"""Top-level Click entry point. Bulky command bodies live in `commands.py`."""

from __future__ import annotations

from pathlib import Path

import click

from roomba_lab.interface.cli import commands
from roomba_lab.shared.config import PROJECT_ROOT


@click.group()
def cli() -> None:
    """roomba-lab — custom DDPG cleaning-robot laboratory."""


@cli.command()
@click.option("--config", type=click.Path(path_type=Path), default=None)
@click.option("--total-timesteps", type=int, default=None)
@click.option("--seed", type=int, default=0)
@click.option("--map-id", type=str, default=None)
@click.option("--save", type=click.Path(path_type=Path), default=None)
def train(config: Path | None, total_timesteps: int | None, seed: int,
          map_id: str | None, save: Path | None) -> None:
    """Train a DDPG agent."""
    commands.cmd_train(config, total_timesteps, seed, map_id, save)


@cli.command()
@click.option("--config", type=click.Path(path_type=Path), default=None)
@click.argument("checkpoint", type=click.Path(exists=True, path_type=Path))
@click.option("--n-episodes", type=int, default=5)
@click.option("--seed", type=int, default=0)
@click.option("--map-id", type=str, default=None)
def evaluate(config: Path | None, checkpoint: Path, n_episodes: int,
             seed: int, map_id: str | None) -> None:
    """Evaluate a saved checkpoint deterministically."""
    commands.cmd_evaluate(config, checkpoint, n_episodes, seed, map_id)


@cli.command(name="download-data")
@click.option("--target", type=click.Path(path_type=Path),
              default=PROJECT_ROOT / "data" / "raw" / "sample_maps")
def download_data(target: Path) -> None:
    """Confirm the 10-map sample shipped in the repo."""
    commands.cmd_download_data(target)


@cli.command(name="record-gif")
@click.option("--config", type=click.Path(path_type=Path), default=None)
@click.argument("checkpoint", type=click.Path(exists=True, path_type=Path))
@click.option("--out", type=click.Path(path_type=Path),
              default=PROJECT_ROOT / "assets" / "gifs" / "cleaning_episode.gif")
@click.option("--seed", type=int, default=0)
@click.option("--map-id", type=str, default=None)
def record_gif(config: Path | None, checkpoint: Path, out: Path,
               seed: int, map_id: str | None) -> None:
    """Record a GIF of one cleaning episode."""
    commands.cmd_record_gif(config, checkpoint, out, seed, map_id)


@cli.command()
@click.option("--config", type=click.Path(path_type=Path), default=None)
@click.argument("kind", type=click.Choice(["noise_sigma", "tau", "target_network"]))
@click.option("--n-seeds", type=int, default=3)
@click.option("--total-timesteps", type=int, default=4000)
def sweep(config: Path | None, kind: str, n_seeds: int, total_timesteps: int) -> None:
    """Run an empirical sweep over noise σ, τ, or target-network strategy."""
    commands.cmd_sweep(config, kind, n_seeds, total_timesteps)


@cli.command()
@click.option("--config", type=click.Path(path_type=Path), default=None)
def graphify(config: Path | None) -> None:
    """Emit the docs/wiki/ Obsidian Vault from src/roomba_lab via AST walk."""
    commands.cmd_graphify(config)


@cli.command()
def gui() -> None:
    """Launch the PyQt6 GUI (training + visualisation tabs)."""
    from roomba_lab.interface.gui.main_window import launch
    launch()


if __name__ == "__main__":
    cli()

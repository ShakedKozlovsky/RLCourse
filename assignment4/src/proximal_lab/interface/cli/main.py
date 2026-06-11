"""proximal-lab CLI — thin Click wrapper over the SDK."""

from __future__ import annotations

from pathlib import Path

import click

from proximal_lab.sdk.sdk import ProximalLab

_DEFAULT_CONFIG = Path("configs/setup.json")


def _sdk(config: Path) -> ProximalLab:
    return ProximalLab(config_path=config)


@click.group()
@click.version_option()
@click.option("--config", type=click.Path(path_type=Path), default=_DEFAULT_CONFIG,
              help="Path to setup.json.")
@click.pass_context
def cli(ctx: click.Context, config: Path) -> None:
    """proximal-lab: PPO + GAE on MuJoCo continuous control."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = config


@cli.command("train")
@click.option("--env-id", type=str, default=None, help="Gym env id (default: config).")
@click.option("--total-timesteps", type=int, default=None)
@click.option("--steps-per-rollout", type=int, default=None)
@click.option("--seed", type=int, default=0)
@click.pass_context
def train(ctx: click.Context, env_id: str | None, total_timesteps: int | None,
          steps_per_rollout: int | None, seed: int) -> None:
    """Train PPO on the chosen env."""
    sdk = _sdk(ctx.obj["config"])
    result = sdk.train_ppo(env_id=env_id, total_timesteps=total_timesteps,
                            steps_per_rollout=steps_per_rollout, seed=seed)
    click.echo(f"final_mean_reward={result.final_mean_reward:.3f} "
                f"iterations={len(result.diagnostics)} "
                f"timesteps={result.total_timesteps}")


@cli.command("evaluate")
@click.option("--env-id", type=str, default=None)
@click.option("--n-episodes", type=int, default=10)
@click.pass_context
def evaluate(ctx: click.Context, env_id: str | None, n_episodes: int) -> None:
    """Evaluate the most-recently-trained policy."""
    sdk = _sdk(ctx.obj["config"])
    # Training first (quick) so eval has something to evaluate when invoked solo.
    sdk.train_ppo(env_id=env_id, total_timesteps=512, steps_per_rollout=256)
    result = sdk.evaluate(env_id=env_id, n_episodes=n_episodes)
    click.echo(f"mean_reward={result.mean_reward:.3f} ± {result.std_reward:.3f} "
                f"over {result.n_episodes} episodes")


@cli.command("graphify")
@click.option("--src-dir", type=click.Path(path_type=Path), default=None)
@click.option("--out-dir", type=click.Path(path_type=Path), default=None)
@click.pass_context
def graphify(ctx: click.Context, src_dir: Path | None, out_dir: Path | None) -> None:
    """Generate the Obsidian-compatible wiki from the project's Python AST."""
    from proximal_lab.tools.graphify.runner import run_graphify

    sdk = _sdk(ctx.obj["config"])
    run_graphify(sdk.config, src_dir=src_dir, out_dir=out_dir)


@cli.command("sweep")
@click.argument("kind", type=click.Choice(["lambda", "gamma", "clip_eps"]))
@click.option("--env-id", type=str, default=None)
@click.option("--timesteps-per-cell", type=int, default=20000)
@click.option("--n-seeds", type=int, default=1)
@click.pass_context
def sweep(ctx: click.Context, kind: str, env_id: str | None,
          timesteps_per_cell: int, n_seeds: int) -> None:
    """Run one of the empirical sweeps: lambda / gamma / clip_eps."""
    from proximal_lab.sdk.experiments import ExperimentService

    sdk = _sdk(ctx.obj["config"])
    svc = ExperimentService(sdk, timesteps_per_cell=timesteps_per_cell, n_seeds=n_seeds)
    report = getattr(svc, f"run_{kind}_sweep")(env_id=env_id)
    out = sdk.config.path("results_dir") / "sweeps" / f"{kind}.json"
    svc.save(report, out)
    click.echo(f"wrote {out} — {len(report.cells)} cells")


@cli.command("gui")
@click.pass_context
def gui(ctx: click.Context) -> None:  # pragma: no cover - launches QApplication
    from proximal_lab.interface.gui.main_window import launch
    launch(config_path=ctx.obj["config"])


if __name__ == "__main__":  # pragma: no cover
    cli(obj={})

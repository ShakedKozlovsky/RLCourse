"""CLI command bodies — bulky handlers kept out of `main.py` to keep that file
under the V3 150-LOC cap."""

from __future__ import annotations

import json
from pathlib import Path

import click

from roomba_lab.sdk.sdk import RoombaLab


def cmd_train(config: Path | None, total_timesteps: int | None, seed: int,
              map_id: str | None, save: Path | None) -> None:
    lab = RoombaLab(config_path=config)
    result = lab.train(total_timesteps=total_timesteps, seed=seed, map_id=map_id)
    click.echo(f"diagnostics: {len(result.diagnostics)}")
    if result.final_metrics is not None:
        m = result.final_metrics
        click.echo(f"final coverage={m.coverage:.3f} reward={m.reward:.2f} "
                    f"collisions={m.collisions}")
    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        # Save diagnostics only (network save handled by GUI / explicit save)
        payload = {
            "diagnostics": [d.__dict__ for d in result.diagnostics],
            "final_metrics": result.final_metrics.__dict__
                              if result.final_metrics else None,
        }
        save.write_text(json.dumps(payload, indent=2))
        click.echo(f"wrote {save}")


def cmd_evaluate(config: Path | None, checkpoint: Path, n_episodes: int,
                  seed: int, map_id: str | None) -> None:
    import torch

    from roomba_lab.model.actor_critic_network import ActorCriticNet
    lab = RoombaLab(config_path=config)
    env = lab.make_env(map_id=map_id)
    net = ActorCriticNet(
        obs_dim=env.obs_dim, action_dim=env.action_dim,
        actor_hidden_sizes=tuple(lab.config.get("ddpg.actor_hidden_sizes")),
        critic_hidden_sizes=tuple(lab.config.get("ddpg.critic_hidden_sizes")),
    )
    net.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    report = lab.evaluate(net, n_episodes=n_episodes, seed=seed, map_id=map_id)
    click.echo(json.dumps(report, indent=2))


def cmd_download_data(target: Path) -> None:
    click.echo(f"10 sample maps already committed in: {target}")


def cmd_record_gif(config: Path | None, checkpoint: Path, out: Path,
                    seed: int, map_id: str | None) -> None:
    from roomba_lab.tools.viz.gif_recorder import record_cleaning_gif
    record_cleaning_gif(config, checkpoint, out, seed=seed, map_id=map_id)
    click.echo(f"wrote {out}")


def cmd_sweep(config: Path | None, kind: str, n_seeds: int,
               total_timesteps: int) -> None:
    from roomba_lab.sdk.experiments import ExperimentService
    lab = RoombaLab(config_path=config)
    svc = ExperimentService(lab, n_seeds=n_seeds, total_timesteps=total_timesteps)
    out = svc.run(kind)
    click.echo(f"wrote {out}")


def cmd_graphify(config: Path | None) -> None:
    from roomba_lab.tools.graphify.runner import run as graphify_run
    lab = RoombaLab(config_path=config)
    out = graphify_run(lab.config)
    click.echo(f"wrote wiki to {out}")

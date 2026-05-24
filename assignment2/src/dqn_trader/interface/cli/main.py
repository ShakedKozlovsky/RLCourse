"""Click-based CLI. The CLI is a thin shell over TradingSDK; no logic lives here."""

from __future__ import annotations

import json
from pathlib import Path

import click

from dqn_trader.sdk.sdk import TradingSDK
from dqn_trader.shared.config import ConfigManager
from dqn_trader.shared.version import __version__


def _sdk(config_path: str | None) -> TradingSDK:
    cfg = ConfigManager(setup_path=Path(config_path)) if config_path else ConfigManager()
    return TradingSDK(cfg)


@click.group()
@click.version_option(__version__)
@click.option("--config", "config_path", default=None, help="Path to setup.json (default: configs/setup.json)")
@click.pass_context
def cli(ctx: click.Context, config_path: str | None) -> None:
    """DQN Trader CLI — data, training, backtest, prediction."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = config_path


@cli.command("data")
@click.option("--ticker", default=None, help="Override the ticker from config.")
@click.pass_context
def cmd_data(ctx: click.Context, ticker: str | None) -> None:
    """Run the data pipeline and print resulting tensor shapes."""
    sdk = _sdk(ctx.obj["config"])
    out = sdk.prepare_data(ticker)
    click.echo(f"train features: {out.train.features.shape}")
    click.echo(f"val   features: {out.val.features.shape}")
    click.echo(f"test  features: {out.test.features.shape}")


@cli.command("train")
@click.option("--ticker", default=None)
@click.pass_context
def cmd_train(ctx: click.Context, ticker: str | None) -> None:
    """Train a DQN agent end-to-end."""
    sdk = _sdk(ctx.obj["config"])
    result = sdk.train(ticker=ticker)
    final = result.metrics[-1]
    click.echo(f"episodes: {len(result.metrics)}  final_val_return: {final.val_return:+.4f}")
    click.echo(f"run dir: {result.run_dir}")


@cli.command("backtest")
@click.option("--checkpoint", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--slice", "slice_name", default="test",
              type=click.Choice(["train", "val", "test"]))
@click.pass_context
def cmd_backtest(ctx: click.Context, checkpoint: Path, slice_name: str) -> None:
    """Run the backtest service on a slice using a checkpoint."""
    sdk = _sdk(ctx.obj["config"])
    result = sdk.backtest(checkpoint, slice_name=slice_name)
    click.echo(json.dumps(result.metrics.__dict__, indent=2))


@cli.command("predict")
@click.option("--checkpoint", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--slice", "slice_name", default="test",
              type=click.Choice(["train", "val", "test"]),
              help="Slice to draw the latest market window from.")
@click.option("--position", type=int, default=0)
@click.option("--pnl", type=float, default=0.0, help="Scaled unrealised PnL (default 0).")
@click.pass_context
def cmd_predict(
    ctx: click.Context, checkpoint: Path, slice_name: str, position: int, pnl: float
) -> None:
    """Single-decision inference on the latest available market window."""
    sdk = _sdk(ctx.obj["config"])
    pipeline = sdk.prepare_data()
    market = getattr(pipeline, slice_name).features[-1]  # last (window, 8)
    decision = sdk.predict(market, checkpoint=checkpoint, position=position, pnl_unrealised_scaled=pnl)
    click.echo(json.dumps({
        "action": decision.action.name,
        "q_values": decision.q_values.tolist(),
        "confidence": decision.confidence,
    }, indent=2))


@cli.command("menu")
@click.pass_context
def cmd_menu(ctx: click.Context) -> None:
    """Interactive numbered menu that loops until the user selects 0 (Exit)."""
    sdk = _sdk(ctx.obj["config"])
    while True:
        click.echo("\n=== DQN Trader — Interactive Menu ===")
        click.echo("1. Prepare data")
        click.echo("2. Train agent")
        click.echo("3. Run backtest")
        click.echo("4. Predict next action")
        click.echo("5. Run experiments")
        click.echo("0. Exit")
        choice = click.prompt("Select an option", type=click.Choice(["0", "1", "2", "3", "4", "5"]))

        if choice == "0":
            click.echo("Goodbye.")
            break

        elif choice == "1":
            out = sdk.prepare_data()
            click.echo(f"train features: {out.train.features.shape}")
            click.echo(f"val   features: {out.val.features.shape}")
            click.echo(f"test  features: {out.test.features.shape}")

        elif choice == "2":
            result = sdk.train()
            final = result.metrics[-1]
            click.echo(f"episodes: {len(result.metrics)}  final_val_return: {final.val_return:+.4f}")
            click.echo(f"run dir: {result.run_dir}")

        elif choice == "3":
            ckpt = click.prompt("Checkpoint path", type=click.Path(exists=True, path_type=Path))
            result = sdk.backtest(ckpt)
            click.echo(json.dumps(result.metrics.__dict__, indent=2))

        elif choice == "4":
            ckpt = click.prompt("Checkpoint path", type=click.Path(exists=True, path_type=Path))
            pipeline = sdk.prepare_data()
            market = pipeline.test.features[-1]
            decision = sdk.predict(market, checkpoint=ckpt)
            click.echo(json.dumps({
                "action": decision.action.name,
                "q_values": decision.q_values.tolist(),
                "confidence": decision.confidence,
            }, indent=2))

        elif choice == "5":
            result = sdk.run_experiments()
            click.echo(json.dumps(result if isinstance(result, dict) else str(result), indent=2))


if __name__ == "__main__":  # pragma: no cover
    cli(obj={})

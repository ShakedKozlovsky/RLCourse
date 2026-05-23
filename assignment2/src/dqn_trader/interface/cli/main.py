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


if __name__ == "__main__":  # pragma: no cover
    cli(obj={})

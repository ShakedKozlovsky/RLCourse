"""Train with the improved config and backtest against baseline.

Changes from baseline:
  - 100 episodes (was 30) — more time to converge
  - window_size=50 (was 30) — our sweep showed 50 is best
  - per.enabled=false (was true) — uniform replay beat PER on this data
  - lr=0.0002 (was 0.0005) — slower learning to reduce overfitting

Usage:
    uv run python scripts/run_improved.py
"""

from __future__ import annotations

import json
from pathlib import Path

from dqn_trader.sdk.sdk import TradingSDK
from dqn_trader.shared.config import ConfigManager
from dqn_trader.shared.logger import get_logger

_logger = get_logger("improved")

IMPROVED_CONFIG = Path("configs/setup_improved.json")
RATE_LIMITS = Path("configs/rate_limits.json")


def main() -> None:
    """Train with improved config, backtest, and print comparison."""
    _logger.info("=== training with improved config ===")
    _logger.info(
        "changes: 100 eps, window=50, uniform replay, lr=2e-4"
    )
    cfg = ConfigManager(
        setup_path=IMPROVED_CONFIG, rate_limits_path=RATE_LIMITS
    )
    sdk = TradingSDK(cfg)
    result = sdk.train()
    _logger.info("training done — run dir: %s", result.run_dir)
    ckpt = result.run_dir / "checkpoints" / "best.pt"
    bt = sdk.backtest(
        ckpt, pipeline=result.pipeline, report_name="improved_backtest"
    )
    m = bt.metrics
    _logger.info("=== improved backtest results ===")
    _logger.info(
        "return=%.2f%%  sharpe=%.2f  max_dd=%.2f%%  "
        "win_rate=%.1f%%  trades=%d",
        m.total_return * 100, m.sharpe, m.max_drawdown * 100,
        m.win_rate * 100, m.n_trades,
    )
    # Save comparison summary
    summary = {
        "config_changes": {
            "episodes": "30 → 100",
            "window_size": "30 → 50",
            "per.enabled": "true → false (uniform replay)",
            "lr": "0.0005 → 0.0002",
        },
        "baseline_metrics": {
            "total_return": -0.2231,
            "sharpe": -3.93,
            "max_drawdown": -0.2409,
            "win_rate": 0.3571,
            "n_trades": 14,
        },
        "improved_metrics": m.__dict__,
    }
    out = Path("results/improvement_comparison.json")
    out.write_text(json.dumps(summary, indent=2, default=str))
    _logger.info("comparison saved to %s", out)


if __name__ == "__main__":
    main()

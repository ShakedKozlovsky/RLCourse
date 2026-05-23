"""Risk metrics for evaluating a trading policy's equity curve.

We avoid pandas here so the metrics file can be imported standalone in
notebooks or tests without dragging the data layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TRADING_DAYS = 252.0


@dataclass(frozen=True)
class BacktestMetrics:
    """Compact summary of a single backtest run."""

    total_return: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    n_trades: int
    final_value: float


def total_return(equity: np.ndarray) -> float:
    """``(V_T − V_0) / V_0``. ``equity`` must have at least 2 points."""
    if equity.size < 2:
        return 0.0
    return float((equity[-1] - equity[0]) / equity[0])


def sharpe_ratio(returns: np.ndarray) -> float:
    """Annualised Sharpe of *daily* simple returns. 0 when stdev is zero."""
    if returns.size < 2:
        return 0.0
    sd = float(returns.std(ddof=0))
    if sd == 0:
        return 0.0
    return float(np.sqrt(TRADING_DAYS) * returns.mean() / sd)


def max_drawdown(equity: np.ndarray) -> float:
    """Largest peak-to-trough drop, expressed as a negative fraction of the peak."""
    if equity.size < 2:
        return 0.0
    peaks = np.maximum.accumulate(equity)
    drawdowns = (equity - peaks) / peaks
    return float(drawdowns.min())


def win_rate(trade_pnls: np.ndarray) -> float:
    """Fraction of closed trades with strictly positive P&L. Returns 0 for no trades."""
    if trade_pnls.size == 0:
        return 0.0
    return float((trade_pnls > 0).sum() / trade_pnls.size)


def summarise(equity: np.ndarray, trade_pnls: np.ndarray) -> BacktestMetrics:
    """Bundle of the four standard metrics + trade count + final equity."""
    rets = np.diff(equity) / equity[:-1] if equity.size >= 2 else np.zeros(0, dtype=np.float64)
    return BacktestMetrics(
        total_return=total_return(equity),
        sharpe=sharpe_ratio(rets),
        max_drawdown=max_drawdown(equity),
        win_rate=win_rate(trade_pnls),
        n_trades=int(trade_pnls.size),
        final_value=float(equity[-1]) if equity.size else 0.0,
    )

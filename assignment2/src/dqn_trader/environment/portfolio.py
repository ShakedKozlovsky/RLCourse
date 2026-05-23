"""Portfolio bookkeeping for the trading environment.

All-in / all-out positioning (PRD §6). Costs and slippage are applied as a
single factor on each trade — this keeps cash, position, and reward
arithmetic consistent (see ADR-008 in PLAN.md).
"""

from __future__ import annotations

from dataclasses import dataclass

from dqn_trader.shared.types import Action


@dataclass
class Trade:
    """Diagnostic record of an executed trade."""

    side: Action  # BUY or SELL
    price: float
    shares: float
    notional: float  # signed: + on buy, − on sell
    cost: float  # total friction paid (cost + slippage)


class Portfolio:
    """Owns cash, position, and shares for a single-asset all-in / all-out agent."""

    def __init__(self, initial_capital: float, alpha: float, beta: float):
        if initial_capital <= 0:
            raise ValueError("initial_capital must be > 0")
        if alpha < 0 or beta < 0:
            raise ValueError("alpha and beta must be non-negative")
        self.initial_capital = float(initial_capital)
        self._alpha = float(alpha)
        self._beta = float(beta)
        self.cash = self.initial_capital
        self.shares = 0.0
        self.entry_price = 0.0

    @property
    def position(self) -> int:
        return 0 if self.shares == 0.0 else 1

    @property
    def friction(self) -> float:
        """Combined cost + slippage factor applied on each trade leg."""
        return self._alpha + self._beta

    def value(self, mark_price: float) -> float:
        """Mark-to-market portfolio value at the given price."""
        return self.cash + self.shares * mark_price

    def unrealized_pnl(self, mark_price: float) -> float:
        """P&L on the open position only (0 when flat)."""
        if self.position == 0:
            return 0.0
        return self.shares * (mark_price - self.entry_price)

    def buy(self, price: float) -> Trade | None:
        """Enter long with all cash at ``price``. No-op if already long."""
        if self.position == 1 or self.cash <= 0.0:
            return None
        notional_after_cost = self.cash * (1.0 - self.friction)
        shares = notional_after_cost / price
        cost = self.cash - notional_after_cost
        self.shares = shares
        self.entry_price = price
        self.cash = 0.0
        return Trade(
            side=Action.BUY, price=price, shares=shares, notional=notional_after_cost, cost=cost
        )

    def sell(self, price: float) -> Trade | None:
        """Liquidate the long position at ``price``. No-op if flat."""
        if self.position == 0:
            return None
        gross = self.shares * price
        net = gross * (1.0 - self.friction)
        cost = gross - net
        trade = Trade(
            side=Action.SELL, price=price, shares=self.shares, notional=-net, cost=cost
        )
        self.cash = net
        self.shares = 0.0
        self.entry_price = 0.0
        return trade

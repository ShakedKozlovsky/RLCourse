"""ExperimentService — runs the comparative experiments mandated by the PRD.

Each method produces an ExperimentResult: a list of ConditionResult, one per
variant in the experiment. The summary is also appended to
``results/experiments_summary.md`` as a Markdown table so it shows up
unmodified in the final README.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dqn_trader.sdk.sdk import TradingSDK
from dqn_trader.services.risk_metrics import BacktestMetrics
from dqn_trader.shared.config import ConfigManager
from dqn_trader.shared.logger import get_logger

_logger = get_logger(__name__)


@dataclass(frozen=True)
class ConditionResult:
    """One variant in a comparative experiment."""

    name: str
    overrides: dict[str, Any]
    metrics: BacktestMetrics
    run_dir: str


@dataclass(frozen=True)
class ExperimentResult:
    """Full payload of a single comparative experiment."""

    name: str
    conditions: list[ConditionResult]


class ExperimentService:
    """Runs comparative experiments by repeated train → backtest with config overrides."""

    def __init__(self, base_config: ConfigManager, *, device: str = "cpu"):
        self._base = base_config
        self._device = device

    def run_dqn_vs_dueling(self) -> ExperimentResult:
        return self._compare(
            "dqn_vs_dueling",
            {"vanilla_dqn": {"agent.dueling": False}, "dueling_dqn": {"agent.dueling": True}},
        )

    def run_uniform_vs_per(self) -> ExperimentResult:
        return self._compare(
            "uniform_vs_per",
            {"uniform_replay": {"per.enabled": False}, "prioritized_replay": {"per.enabled": True}},
        )

    def run_reward_variants(self) -> ExperimentResult:
        return self._compare(
            "reward_variants",
            {"baseline": {"env.reward_variant": "baseline"},
             "risk_adjusted": {"env.reward_variant": "risk_adjusted"}},
        )

    def run_cross_ticker(self, secondary_ticker: str | None = None) -> ExperimentResult:
        primary = str(self._base.get("data.ticker"))
        secondary = secondary_ticker or str(self._base.get("data.comparative_ticker", "SPY"))
        return self._compare(
            "cross_ticker",
            {primary: {"data.ticker": primary}, secondary: {"data.ticker": secondary}},
        )

    def _compare(self, name: str, conditions: dict[str, dict[str, Any]]) -> ExperimentResult:
        _logger.info("experiment: %s — %d conditions", name, len(conditions))
        results: list[ConditionResult] = []
        for cond_name, overrides in conditions.items():
            sdk = self._build_sdk(overrides)
            train = sdk.train()
            ckpt = train.run_dir / "checkpoints" / "best.pt"
            bt = sdk.backtest(ckpt, slice_name="test", pipeline=train.pipeline)
            results.append(ConditionResult(
                name=cond_name, overrides=overrides, metrics=bt.metrics,
                run_dir=str(train.run_dir),
            ))
        experiment = ExperimentResult(name=name, conditions=results)
        self._append_markdown(experiment)
        return experiment

    def _build_sdk(self, overrides: dict[str, Any]) -> TradingSDK:
        merged = self._merge(copy.deepcopy(dict(self._base.setup)), overrides)
        tmp_path = Path(self._base.path("results_dir")) / f"_tmp_config_{id(overrides)}.json"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(json.dumps(merged))
        cfg = ConfigManager(setup_path=tmp_path)
        return TradingSDK(cfg, device=self._device)

    @staticmethod
    def _merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
        for dotted, value in overrides.items():
            node = base
            parts = dotted.split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
        return base

    def _append_markdown(self, exp: ExperimentResult) -> None:
        out = Path(self._base.path("results_dir")) / "experiments_summary.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"\n## {exp.name}\n",
                 "| condition | overrides | total_return | sharpe | max_dd | win_rate | n_trades |",
                 "|---|---|---|---|---|---|---|"]
        for c in exp.conditions:
            ov = ", ".join(f"`{k}={v}`" for k, v in c.overrides.items())
            m = c.metrics
            lines.append(
                f"| {c.name} | {ov} | {m.total_return:+.3%} | {m.sharpe:.2f} | "
                f"{m.max_drawdown:.2%} | {m.win_rate:.2%} | {m.n_trades} |"
            )
        out.open("a", encoding="utf-8").write("\n".join(lines) + "\n")
        json_out = out.parent / f"{exp.name}.json"
        json_out.write_text(json.dumps({"name": exp.name,
                                         "conditions": [asdict(c) for c in exp.conditions]},
                                        indent=2, default=str))

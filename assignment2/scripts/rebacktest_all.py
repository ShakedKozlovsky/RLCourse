"""Re-run backtests on existing checkpoints with unique report names.

Why: the first run of ExperimentService used the pre-patch SDK that
saved every backtest to `test_backtest.npz`, so each condition's curves
overwrote the previous. The patched SDK accepts `report_name` so we
can re-emit all 8 backtests with stable names — no retraining needed.
"""

from __future__ import annotations

import json
from pathlib import Path

from dqn_trader.services.experiment_service import ExperimentService
from dqn_trader.shared.config import ConfigManager
from dqn_trader.shared.logger import get_logger

_logger = get_logger("rebacktest")

EXPERIMENTS = {
    "dqn_vs_dueling": {
        "vanilla_dqn": {"agent.dueling": False},
        "dueling_dqn": {"agent.dueling": True},
    },
    "uniform_vs_per": {
        "uniform_replay": {"per.enabled": False},
        "prioritized_replay": {"per.enabled": True},
    },
    "reward_variants": {
        "baseline": {"env.reward_variant": "baseline"},
        "risk_adjusted": {"env.reward_variant": "risk_adjusted"},
    },
    "cross_ticker": {
        "AAPL": {"data.ticker": "AAPL"},
        "SPY": {"data.ticker": "SPY"},
    },
}


def main() -> None:
    cfg = ConfigManager()
    exp_svc = ExperimentService(cfg)
    for exp_name, conditions in EXPERIMENTS.items():
        payload = json.loads((Path("results") / f"{exp_name}.json").read_text())
        for entry, (cond_name, overrides) in zip(payload["conditions"], conditions.items(), strict=True):
            assert entry["name"] == cond_name, f"{entry['name']} != {cond_name}"
            ckpt = Path(entry["run_dir"]) / "checkpoints" / "best.pt"
            sdk = exp_svc._build_sdk(overrides)
            sdk.backtest(ckpt, slice_name="test", report_name=f"{exp_name}__{cond_name}")
            _logger.info("rebacktest: %s :: %s", exp_name, cond_name)


if __name__ == "__main__":
    main()

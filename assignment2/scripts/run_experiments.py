"""Driver that runs all four comparative experiments end-to-end.

Use:
    uv run python scripts/run_experiments.py
"""

from __future__ import annotations

from dqn_trader.services.experiment_service import ExperimentService
from dqn_trader.shared.config import ConfigManager
from dqn_trader.shared.logger import get_logger


def main() -> None:
    log = get_logger("driver")
    cfg = ConfigManager()
    exp = ExperimentService(cfg)
    log.info("=== dqn_vs_dueling ===")
    exp.run_dqn_vs_dueling()
    log.info("=== uniform_vs_per ===")
    exp.run_uniform_vs_per()
    log.info("=== reward_variants ===")
    exp.run_reward_variants()
    log.info("=== cross_ticker ===")
    exp.run_cross_ticker()
    log.info("done — see results/experiments_summary.md")


if __name__ == "__main__":
    main()

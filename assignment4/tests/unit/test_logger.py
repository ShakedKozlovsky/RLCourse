"""Logger factory — single configuration point."""

from __future__ import annotations

import logging

from proximal_lab.shared.logger import get_logger


def test_returns_named_logger() -> None:
    log = get_logger("test.module")
    assert isinstance(log, logging.Logger)
    assert log.name == "test.module"


def test_subsequent_calls_share_root_config() -> None:
    """First call configures the root handler; second call must not duplicate it."""
    get_logger("first")
    n_handlers = len(logging.getLogger().handlers)
    get_logger("second")
    assert len(logging.getLogger().handlers) == n_handlers


def test_logger_emits_info(caplog) -> None:  # noqa: ANN001
    log = get_logger("emits.info")
    with caplog.at_level(logging.INFO, logger="emits.info"):
        log.info("hello %s", "world")
    assert any("hello world" in r.message for r in caplog.records)

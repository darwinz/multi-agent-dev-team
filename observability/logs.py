"""JSON-lines event logging with run/agent context and secret redaction."""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from observability.context import current_agent, current_run

LOGGER_NAME = "mad"
_SECRET_NAME = re.compile(r"(_API_KEY|_TOKEN|_SECRET)$")

# Let INFO events through even when configure_logging was never called.
logging.getLogger(LOGGER_NAME).setLevel(logging.INFO)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        run = current_run()
        out = dict(getattr(record, "fields", {}))
        out.update(
            ts=datetime.fromtimestamp(record.created, UTC).isoformat(),
            level=record.levelname,
            run_id=getattr(record, "run_id", run.run_id if run else None),
            agent=getattr(record, "agent", current_agent()),
            event=record.getMessage(),
        )
        return json.dumps(out, default=str)


def redact(text: str) -> str:
    secrets = [v for k, v in os.environ.items() if _SECRET_NAME.search(k) and len(v) >= 8]
    for v in sorted(secrets, key=len, reverse=True):
        text = text.replace(v, "[redacted]")
    return text


def configure_logging(level: str = "INFO") -> None:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level.upper())
    logger.propagate = True  # so caplog sees records
    if not any(getattr(h, "_mad_stderr", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        handler._mad_stderr = True  # type: ignore[attr-defined]
        logger.addHandler(handler)


def log_event(event: str, *, level: int = logging.INFO, **fields) -> None:
    run = current_run()
    clean = {k: redact(v) if isinstance(v, str) else v for k, v in fields.items()}
    logging.getLogger(LOGGER_NAME).log(
        level, event,
        extra={"fields": clean, "run_id": run.run_id if run else None, "agent": current_agent()})


@contextmanager
def run_log_file(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger(LOGGER_NAME)
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        handler.close()

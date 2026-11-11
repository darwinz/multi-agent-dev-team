"""JSON-lines event logging with run/agent context and secret redaction."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
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
    """Formats log records as one JSON line with timestamp, level, run id, agent and event, with secrets redacted."""

    def format(self, record: logging.LogRecord) -> str:
        """Render the record's fields plus ts, level, run_id, agent and event as redacted JSON."""
        run = current_run()
        out = dict(getattr(record, "fields", {}))
        out.update(
            ts=datetime.fromtimestamp(record.created, UTC).isoformat(),
            level=record.levelname,
            run_id=getattr(record, "run_id", run.run_id if run else None),
            agent=getattr(record, "agent", current_agent()),
            event=record.getMessage(),
        )
        return redact(json.dumps(out, default=str))


class _StderrHandler(logging.StreamHandler):
    """Writes to whatever sys.stderr is at emit time, so a replaced stderr (pytest capture) never goes stale."""

    @property
    def stream(self):  # type: ignore[override]
        return sys.stderr

    @stream.setter
    def stream(self, value) -> None:  # StreamHandler.__init__ assigns it; the property ignores that
        pass


def redact(text: str) -> str:
    """Replace the value of any *_API_KEY, *_TOKEN or *_SECRET environment variable (8+ chars) with [redacted]."""
    secrets = [v for k, v in os.environ.items() if _SECRET_NAME.search(k) and len(v) >= 8]
    for v in sorted(secrets, key=len, reverse=True):
        text = text.replace(v, "[redacted]")
    return text


def configure_logging(level: str = "INFO") -> None:
    """Set the `mad` logger's level and attach one stderr JSON handler (idempotent)."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level.upper())
    logger.propagate = True  # so caplog sees records
    if not any(getattr(h, "_mad_stderr", False) for h in logger.handlers):
        handler = _StderrHandler()
        handler.setFormatter(JsonFormatter())
        handler._mad_stderr = True  # type: ignore[attr-defined]
        logger.addHandler(handler)


def error_class(e: BaseException) -> str:
    """The class name of what actually failed: resilience wrappers (AgentError) carry the original."""
    return type(getattr(e, "original", None) or e).__name__


def log_event(event: str, *, level: int = logging.INFO, **fields) -> None:
    """Log a named event with structured fields (string values redacted) tagged with the current run and agent."""
    run = current_run()
    clean = {k: redact(v) if isinstance(v, str) else v for k, v in fields.items()}
    logging.getLogger(LOGGER_NAME).log(
        level, event,
        extra={"fields": clean, "run_id": run.run_id if run else None, "agent": current_agent()})


@contextmanager
def run_log_file(path: Path) -> Iterator[None]:
    """Write this run's log records (as JSON lines) to path for the duration of the with-block."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(JsonFormatter())
    run = current_run()
    if run is not None:  # only this run's records; overlapping runs share the process-global logger
        run_id = run.run_id
        handler.addFilter(lambda record: getattr(record, "run_id", None) == run_id)
    logger = logging.getLogger(LOGGER_NAME)
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        handler.close()

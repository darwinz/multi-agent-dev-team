"""Every tool call goes through here: bounded retry, the shared "tools" breaker and a ToolRecord."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TypeVar

from observability.context import current_agent, record_tool
from observability.costs import ToolRecord
from observability.logs import log_event
from resilience.errors import PermanentError, TransientError, classify_tool_error
from resilience.runtime import resilient_call

T = TypeVar("T")


def guarded_tool(name: str, fn: Callable[[], T]) -> T:
    """Run fn under retry + the "tools" breaker. A permanent failure re-raises the tool's own exception,
    so callers' except-clauses see what they saw before; a TransientError (incl. CircuitOpenError)
    propagates as-is."""
    start = time.monotonic()
    ok, attempts = False, 1
    try:
        result, attempts = resilient_call(fn, breaker="tools", classify=classify_tool_error, name=name)
        ok = True
        return result
    except TransientError as e:
        attempts = e.attempts
        raise
    except PermanentError as e:
        if e.original is not None and e.__cause__ is e.original:  # wrapped by call_with_retry
            raise e.original from None
        raise
    finally:
        duration_ms = (time.monotonic() - start) * 1000
        record_tool(ToolRecord(ts=datetime.now(UTC).isoformat(), agent=current_agent(), tool=name, ok=ok,
                               duration_ms=duration_ms, attempts=attempts))
        log_event("tool_call", level=logging.INFO if ok else logging.WARNING, tool=name, ok=ok,
                  duration_ms=round(duration_ms, 1), attempts=attempts)

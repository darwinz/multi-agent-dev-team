"""Per-run and per-agent context carried in ContextVars so threads and logs can see it."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # defined in Task 6
    from observability.costs import CallRecord, ToolRecord


@dataclass
class RunContext:
    """Mutable record of one run: id, collected call and tool records (guarded by lock) and trace URL."""

    run_id: str
    calls: list[CallRecord] = field(default_factory=list)
    tools: list[ToolRecord] = field(default_factory=list)
    trace_url: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


_run: ContextVar[RunContext | None] = ContextVar("mad_run", default=None)
_agent: ContextVar[str | None] = ContextVar("mad_agent", default=None)


def _reset(var: ContextVar, token, prev) -> None:
    """Restore var; tolerate teardown from a different Context (generator-held scopes)."""
    try:
        var.reset(token)
    except ValueError:
        var.set(prev)


@contextmanager
def run_scope(ctx: RunContext) -> Iterator[RunContext]:
    """Make ctx the current run for the duration of the with-block, then restore the previous one."""
    prev = _run.get()
    token = _run.set(ctx)
    try:
        yield ctx
    finally:
        _reset(_run, token, prev)


def current_run() -> RunContext | None:
    """The active RunContext, or None outside a run_scope."""
    return _run.get()


@contextmanager
def agent_scope(name: str) -> Iterator[None]:
    """Label the with-block's model calls and logs with the agent name, then restore the previous label."""
    prev = _agent.get()
    token = _agent.set(name)
    try:
        yield
    finally:
        _reset(_agent, token, prev)


def current_agent() -> str | None:
    """The active agent name, or None outside an agent_scope."""
    return _agent.get()


def record_call(rec: CallRecord) -> None:
    """Append a model-call record to the current run (thread-safe); no-op outside a run."""
    ctx = current_run()
    if ctx is not None:
        with ctx.lock:
            ctx.calls.append(rec)


def record_tool(rec: ToolRecord) -> None:
    """Append a tool-call record to the current run (thread-safe); no-op outside a run."""
    ctx = current_run()
    if ctx is not None:
        with ctx.lock:
            ctx.tools.append(rec)

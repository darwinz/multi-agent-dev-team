"""LangSmith tracing: one root trace per team run, agent node spans with their span metadata (spec §5.2)."""

from __future__ import annotations

import functools
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal

import langsmith
from langsmith.run_helpers import get_current_run_tree
from langsmith.utils import tracing_is_enabled

from observability.context import agent_scope, current_run
from observability.costs import CallRecord, ToolRecord
from observability.logs import log_event

if TYPE_CHECKING:
    from orchestration.project_state import ProjectState

Agent = Literal["pm", "coder", "qa"]


def tracing_enabled() -> bool:
    """LangSmith tracing is requested and has a key; otherwise tracing is a silent no-op."""
    return bool(tracing_is_enabled()) and bool(os.environ.get("LANGSMITH_API_KEY"))


@contextmanager
def root_trace(run_id: str, requirement: str) -> Iterator[str | None]:
    """The run's one root trace; yields its URL (None when tracing is off or the URL can't be resolved).

    LangGraph's own runs nest under it through LangSmith's context propagation.
    """
    if not tracing_enabled():
        yield None
        return
    with langsmith.trace(name=f"team-run:{run_id}", run_type="chain", inputs={"requirement": requirement},
                         metadata={"run_id": run_id}) as rt:
        try:
            url: str | None = rt.get_url()
        except Exception:  # the URL is a convenience; a lookup failure must not fail the run
            url = None
        yield url


def span_metadata(agent: str, calls: list[CallRecord], tools: list[ToolRecord], duration_ms: float) -> dict:
    """The agent span's fields, from the same per-call records that feed the cost report."""
    return {"agent_name": agent,
            "token_count": sum(c.prompt_tokens + c.completion_tokens for c in calls),
            "tool_calls": len(tools),
            "tool_names": sorted({t.tool for t in tools}),
            "duration_ms": round(duration_ms, 1)}


def agent_node(agent: Agent, fn: Callable[[ProjectState], dict]) -> Callable[[ProjectState], dict]:
    """Run a graph node as `agent`, and attach its span metadata to the node's LangSmith run."""

    @functools.wraps(fn)
    def node(state: ProjectState) -> dict:
        ctx = current_run()
        n_calls, n_tools = (len(ctx.calls), len(ctx.tools)) if ctx is not None else (0, 0)
        started = time.perf_counter()
        with agent_scope(agent):
            try:
                return fn(state)
            finally:  # also on exception: the failed span still says what it spent
                duration_ms = (time.perf_counter() - started) * 1000
                if ctx is not None:
                    with ctx.lock:
                        calls, tools = ctx.calls[n_calls:], ctx.tools[n_tools:]
                else:
                    calls, tools = [], []
                meta = span_metadata(agent, calls, tools, duration_ms)
                # Inside a LangGraph node the current run tree is the node's own run (LangGraph sets it as
                # the tracing parent around the node function); it is patched with this metadata on node end.
                rt = get_current_run_tree()
                if rt is not None:
                    extra = rt.extra if rt.extra is not None else {}
                    extra["metadata"] = {**(extra.get("metadata") or {}), **meta}  # fresh dict: never shared
                    rt.extra = extra
                log_event("agent_end", **meta)

    return node

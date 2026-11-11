"""LangGraph state for one Coder run."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class WriteInfo(TypedDict):
    """Record of a file write: its sequence number and sha256."""

    seq: int
    sha256: str


class ExecInfo(TypedDict):
    """Record of one code execution: sequence, target, outcome, output and the sha256 of each .py file it loaded."""

    seq: int
    target: str
    ok: bool
    exit_code: int | None
    timed_out: bool
    stdout: str
    stderr: str
    duration_s: float
    loaded: dict[str, str]


class ToolEvent(TypedDict):
    """Record of one tool call: sequence, iteration, name, arguments, observation and success."""

    seq: int
    iteration: int
    name: str
    args: dict
    observation: str
    ok: bool


def merge_dicts(left: dict, right: dict) -> dict:
    """Reducer: merge two dicts, right-hand keys win."""
    return {**left, **right}


def merge_hits(left: list[dict], right: list[dict]) -> list[dict]:
    """Reducer: append the right-hand memory hits whose id is not already in the left-hand list."""
    seen = {h["id"] for h in left}
    return left + [h for h in right if h["id"] not in seen]


class CoderState(TypedDict):
    """LangGraph state of one Coder run: task, messages, plan, tool events, writes, executions, memory, result."""

    task: str
    messages: Annotated[list[AnyMessage], add_messages]
    iterations: int
    seq: int  # monotonic counter ordering writes and executions
    plan: list[str]
    events: Annotated[list[ToolEvent], operator.add]
    writes: Annotated[dict[str, WriteInfo], merge_dicts]
    executions: Annotated[list[ExecInfo], operator.add]
    memory_context: str
    memories: Annotated[list[dict], merge_hits]
    decision: dict | None
    finish: dict | None
    stop_reason: Literal["finished", "iteration_limit"] | None
    result: dict | None

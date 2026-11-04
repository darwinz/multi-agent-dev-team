"""Step events streamed to the CLI and web UI while a run progresses."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

EventKind = Literal["start", "memory", "thought", "plan", "action", "observation", "nudge", "limit", "result", "error"]


class StepEvent(BaseModel):
    kind: EventKind
    iteration: int
    max_iterations: int
    data: dict = {}


class TeamEvent(BaseModel):
    """A team-run event: run milestones from TeamRunner, a Coder StepEvent tagged with its task,
    a self-reflection record, a QA round's report, or an A2A broker record."""

    kind: Literal["start", "spec", "consolidation", "handoff", "coder", "task_done", "final", "progress",
                  "reflection", "qa_report", "a2a"]
    task_id: str | None = None
    data: dict = {}

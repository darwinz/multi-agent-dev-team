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
    """A team-graph event: a Coder StepEvent tagged with its task, or a progress note."""

    kind: Literal["coder", "progress"]
    task_id: str | None = None
    data: dict = {}

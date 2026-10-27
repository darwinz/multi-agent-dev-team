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

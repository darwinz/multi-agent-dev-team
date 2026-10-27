"""Routing after each model call (spec §6.1 precedence, top wins)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from langchain_core.messages import AIMessage
from pydantic import ValidationError

from tools.specs import FINISH, FinishArgs

FINISH_MIXED = "finish must be called alone, after your last action"
_PLAN_LINE = re.compile(r"^\s*\d+[.)]\s+(.+)", re.MULTILINE)


@dataclass
class Decision:
    kind: Literal["finish", "execute", "nudge", "limit"]
    run_calls: list[dict] = field(default_factory=list)
    rejections: list[tuple[str, str]] = field(default_factory=list)
    finish_args: dict | None = None


def classify(message: AIMessage, iterations: int, max_iterations: int) -> Decision:
    """`iterations` already counts this call. Calls at the limit are never executed."""
    calls = list(message.tool_calls)
    at_limit = iterations >= max_iterations
    finishes = [c for c in calls if c["name"] == FINISH]
    others = [c for c in calls if c["name"] != FINISH]

    # 1. finish alone
    rejections: list[tuple[str, str]] = []
    if finishes and not others and len(finishes) == 1:
        try:
            args = FinishArgs.model_validate(finishes[0]["args"])
            return Decision("finish", finish_args=args.model_dump())
        except ValidationError as e:
            rejections.append((finishes[0]["id"], f"finish rejected: {e}"))
    else:
        # 2. finish mixed with other calls (or called twice)
        rejections += [(c["id"], FINISH_MIXED) for c in finishes]

    # 3. ordinary calls
    if others:
        if at_limit:
            return Decision("limit", rejections=rejections)
        return Decision("execute", run_calls=others, rejections=rejections)

    # 4. no usable tool calls
    return Decision("limit" if at_limit else "nudge", rejections=rejections)


def extract_plan(text: str) -> list[str]:
    return [m.strip() for m in _PLAN_LINE.findall(text)]

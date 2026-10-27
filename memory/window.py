"""Sliding-window conversation memory: the last N (task, final answer) turns."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Turn:
    task: str
    answer: str


class SlidingWindow:
    def __init__(self, max_turns: int) -> None:
        self._turns: deque[Turn] = deque(maxlen=max_turns)

    def add(self, task: str, answer: str) -> None:
        self._turns.append(Turn(task, answer))

    def turns(self) -> list[Turn]:
        return list(self._turns)

    def clear(self) -> None:
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)

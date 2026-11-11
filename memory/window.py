"""Sliding-window conversation memory: the last N (task, final answer) turns."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Turn:
    """One finished conversation turn: the user's task and the agent's final answer."""

    task: str
    answer: str


class SlidingWindow:
    """Short-term memory: keeps only the most recent max_turns turns, dropping the oldest."""

    def __init__(self, max_turns: int) -> None:
        """Create an empty window holding at most max_turns turns."""
        self._turns: deque[Turn] = deque(maxlen=max_turns)

    def add(self, task: str, answer: str) -> None:
        """Append a turn, evicting the oldest if the window is full."""
        self._turns.append(Turn(task, answer))

    def turns(self) -> list[Turn]:
        """The turns currently held, oldest first, as a new list."""
        return list(self._turns)

    def clear(self) -> None:
        """Forget every turn."""
        self._turns.clear()

    def __len__(self) -> int:
        """Number of turns currently held."""
        return len(self._turns)

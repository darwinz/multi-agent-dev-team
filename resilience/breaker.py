"""Per-name circuit breakers: only exhausted TransientErrors count, one half-open probe at a time."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from enum import StrEnum
from typing import TypeVar

from resilience.errors import CircuitOpenError, TransientError
from resilience.retry import _log  # Task 5 rewires this shim to observability.logs.log_event

T = TypeVar("T")


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(self, name: str, *, failure_threshold: int = 3, cooldown_s: float = 30.0,
                 clock: Callable[[], float] = time.monotonic):
        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self.clock = clock
        self.failures = 0
        self._state = BreakerState.CLOSED
        self._opened_at = 0.0
        self._probing = False
        self._lock = threading.Lock()

    @property
    def state(self) -> BreakerState:
        """OPEN reports as HALF_OPEN once the cool-down has elapsed."""
        if self._state is BreakerState.OPEN and self.clock() - self._opened_at >= self.cooldown_s:
            return BreakerState.HALF_OPEN
        return self._state

    def _transition(self, to: BreakerState) -> None:  # caller holds the lock
        _log("breaker_transition", level=logging.WARNING, breaker=self.name,
             from_state=str(self._state), to_state=str(to), failures=self.failures)
        self._state = to

    def call(self, fn: Callable[[], T]) -> T:
        with self._lock:
            state = self.state
            if state is BreakerState.OPEN or (state is BreakerState.HALF_OPEN and self._probing):
                raise CircuitOpenError(f"circuit '{self.name}' is open", breaker=self.name)
            probe = state is BreakerState.HALF_OPEN
            if probe:
                self._probing = True
                self._transition(BreakerState.HALF_OPEN)
        try:
            result = fn()
        except TransientError:
            with self._lock:
                self.failures += 1
                if probe or self.failures >= self.failure_threshold:
                    self._opened_at = self.clock()
                    self._transition(BreakerState.OPEN)
            raise
        except BaseException:
            raise  # permanent/parsing/KeyboardInterrupt: count and state unchanged
        else:
            with self._lock:
                self.failures = 0
                if probe:
                    self._transition(BreakerState.CLOSED)
            return result
        finally:
            if probe:
                with self._lock:
                    self._probing = False


class BreakerRegistry:
    def __init__(self, *, failure_threshold: int = 3, cooldown_s: float = 30.0,
                 clock: Callable[[], float] = time.monotonic):
        self._threshold = failure_threshold
        self._cooldown_s = cooldown_s
        self._clock = clock
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def get(self, name: str) -> CircuitBreaker:
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(
                    name, failure_threshold=self._threshold, cooldown_s=self._cooldown_s,
                    clock=self._clock)
            return self._breakers[name]

    def configure(self, *, failure_threshold: int, cooldown_s: float,
                  clock: Callable[[], float] = time.monotonic) -> None:
        with self._lock:
            self._threshold, self._cooldown_s, self._clock = failure_threshold, cooldown_s, clock
            for b in self._breakers.values():
                b.failure_threshold, b.cooldown_s, b.clock = failure_threshold, cooldown_s, clock

    def reset(self) -> None:
        with self._lock:
            self._breakers.clear()

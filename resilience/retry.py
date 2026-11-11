"""Call-level retry with full-jitter exponential backoff."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from observability.logs import log_event
from resilience.errors import CircuitOpenError, ErrorKind, PermanentError, TransientError

if TYPE_CHECKING:
    from agents.config import Settings

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """How many retries to make and the bounds of the exponential backoff between them."""

    max_retries: int = 3
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0

    def delay(self, attempt: int, rng: Callable[[float, float], float] = random.uniform) -> float:
        """Full jitter: uniform(0, min(max_delay, base * 2**attempt)), attempt 0-based."""
        return rng(0.0, min(self.max_delay_s, self.base_delay_s * 2 ** attempt))

    @classmethod
    def from_settings(cls, s: Settings) -> RetryPolicy:
        """Build the policy from the retry_* settings."""
        return cls(s.retry_max, s.retry_base_delay_s, s.retry_max_delay_s)


def call_with_retry(fn: Callable[[], T], *, policy: RetryPolicy,
                    classify: Callable[[BaseException], ErrorKind], name: str,
                    sleep: Callable[[float], None] = time.sleep,
                    rng: Callable[[float, float], float] = random.uniform) -> tuple[T, int]:
    """Run fn, retrying transient failures; returns (result, attempts) counting the original call."""
    attempt = 0
    while True:
        try:
            return fn(), attempt + 1
        except Exception as e:
            if isinstance(e, CircuitOpenError):
                raise  # breaker open: fail fast, no retry, no sleep
            if classify(e) == "permanent":
                if isinstance(e, PermanentError):
                    raise
                raise PermanentError(original=e) from e
            if attempt >= policy.max_retries:
                log_event("retry_exhausted", level=logging.WARNING, name=name, attempt=attempt + 1,
                      error_class=type(e).__name__)
                raise TransientError(original=e, attempts=attempt + 1) from e
            delay = policy.delay(attempt, rng)
            log_event("retry_attempt", level=logging.WARNING, name=name, attempt=attempt + 1,
                  error_class=type(e).__name__, delay_s=delay)
            sleep(delay)
            attempt += 1

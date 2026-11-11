"""Process-wide resilience runtime: retry policy, breaker registry and sleep function."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar

from resilience.breaker import BreakerRegistry
from resilience.errors import ErrorKind
from resilience.retry import RetryPolicy, call_with_retry

if TYPE_CHECKING:
    from agents.config import Settings

T = TypeVar("T")


@dataclass
class Resilience:
    """The process-wide retry policy, breaker registry and sleep function."""

    policy: RetryPolicy = field(default_factory=RetryPolicy)
    breakers: BreakerRegistry = field(default_factory=BreakerRegistry)
    sleep: Callable[[float], None] = time.sleep


_runtime = Resilience()


def runtime() -> Resilience:
    """The current process-wide Resilience."""
    return _runtime


def configure(settings: Settings) -> None:
    """Apply the retry and breaker settings to the process-wide runtime."""
    _runtime.policy = RetryPolicy.from_settings(settings)
    _runtime.breakers.configure(failure_threshold=settings.breaker_failure_threshold,
                                cooldown_s=settings.breaker_cooldown_s)


def reset() -> None:
    """Defaults with zero retry delays and a fresh registry (tests)."""
    global _runtime
    _runtime = Resilience(policy=RetryPolicy(base_delay_s=0.0, max_delay_s=0.0),
                          breakers=BreakerRegistry(), sleep=time.sleep)


def resilient_call(fn: Callable[[], T], *, breaker: str,
                   classify: Callable[[BaseException], ErrorKind], name: str) -> tuple[T, int]:
    """Run fn with retry inside the named breaker; returns (result, attempts)."""
    rt = runtime()
    return rt.breakers.get(breaker).call(
        lambda: call_with_retry(fn, policy=rt.policy, classify=classify, name=name, sleep=rt.sleep))

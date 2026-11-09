"""Error classification and call-level retry with full jitter."""

import errno

import pytest

from resilience.errors import (
    CircuitOpenError,
    PermanentError,
    TransientError,
    classify_chroma_error,
    classify_llm_error,
    classify_tool_error,
)
from resilience.retry import RetryPolicy, call_with_retry
from tools.workspace import ToolError


class Flaky(Exception):
    status_code = 429


def counting(fails: int, exc=Flaky):
    calls = {"n": 0}
    def fn():
        calls["n"] += 1
        if calls["n"] <= fails:
            raise exc("busy")
        return "ok"
    return fn, calls

P = RetryPolicy(max_retries=3, base_delay_s=0, max_delay_s=0)

def test_transient_retried_then_succeeds():
    fn, calls = counting(2)
    assert call_with_retry(fn, policy=P, classify=classify_llm_error, name="t", sleep=lambda s: None) == ("ok", 3)

def test_persistent_transient_stops_after_exactly_3_retries():
    fn, calls = counting(99)
    with pytest.raises(TransientError) as ei:
        call_with_retry(fn, policy=P, classify=classify_llm_error, name="t", sleep=lambda s: None)
    assert calls["n"] == 4 and ei.value.attempts == 4 and isinstance(ei.value.original, Flaky)

def test_permanent_not_retried():
    fn, calls = counting(99, exc=ValueError)
    with pytest.raises(PermanentError):
        call_with_retry(fn, policy=P, classify=classify_llm_error, name="t", sleep=lambda s: None)
    assert calls["n"] == 1

def test_full_jitter_bounds():
    p = RetryPolicy(max_retries=3, base_delay_s=1.0, max_delay_s=3.0)
    assert p.delay(0, rng=lambda a, b: b) == 1.0 and p.delay(1, rng=lambda a, b: b) == 2.0
    assert p.delay(5, rng=lambda a, b: b) == 3.0 and p.delay(2, rng=lambda a, b: a) == 0.0

def test_sleeps_use_policy_delays():
    slept = []
    fn, _ = counting(3)
    call_with_retry(fn, policy=RetryPolicy(3, 1.0, 30.0), classify=classify_llm_error, name="t",
                    sleep=slept.append, rng=lambda a, b: b)
    assert slept == [1.0, 2.0, 4.0]

def test_keyboard_interrupt_during_sleep_propagates():  # Review Focus 5
    fn, calls = counting(5)
    def sleep(_):
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        call_with_retry(fn, policy=P, classify=classify_llm_error, name="t", sleep=sleep)
    assert calls["n"] == 1

@pytest.mark.parametrize("exc,kind", [
    (TimeoutError(), "transient"), (ConnectionError(), "transient"), (Flaky(), "transient"),
    (type("OverloadedError", (Exception,), {})(), "transient"), (ValueError("bad"), "permanent"),
    (type("E", (Exception,), {"status_code": 401})(), "permanent"),
    (type("E", (Exception,), {"status_code": 529})(), "transient"),
])
def test_classify_llm_error(exc, kind):
    assert classify_llm_error(exc) == kind

def test_classify_tool_error():
    assert classify_tool_error(OSError(errno.EBUSY, "busy")) == "transient"
    assert classify_tool_error(OSError(errno.ENOENT, "gone")) == "permanent"
    assert classify_tool_error(ToolError("bad path")) == "permanent"

def test_classify_chroma_error():
    assert classify_chroma_error(ConnectionError()) == "transient"
    assert classify_chroma_error(Flaky()) == "transient"
    assert classify_chroma_error(ValueError("bad")) == "permanent"

def test_error_str_keeps_original_text_and_circuit_open_reraised():
    assert str(PermanentError(original=RuntimeError("boom"))) == "RuntimeError: boom"
    def fn():
        raise CircuitOpenError("open", breaker="pm")
    with pytest.raises(CircuitOpenError):
        call_with_retry(fn, policy=P, classify=classify_llm_error, name="t", sleep=lambda s: None)

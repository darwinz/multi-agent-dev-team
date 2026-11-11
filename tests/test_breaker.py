import logging

import pytest

from resilience import runtime
from resilience.breaker import BreakerRegistry, BreakerState, CircuitBreaker
from resilience.errors import CircuitOpenError, PermanentError, TransientError, classify_llm_error
from resilience.retry import RetryPolicy

t = [0.0]


def clock():
    return t[0]


def fail():
    raise TransientError("x")


def ok():
    return "ok"


def _perm():
    raise PermanentError("nope")


def _val():
    raise ValueError("bad")


@pytest.fixture(autouse=True)
def _reset_clock():
    t[0] = 0.0


def _breaker(**kw):
    return CircuitBreaker("pm", failure_threshold=kw.pop("failure_threshold", 3),
                          cooldown_s=kw.pop("cooldown_s", 30), clock=clock)


def _open(b):
    for _ in range(3):
        with pytest.raises(TransientError):
            b.call(fail)
    assert b.state is BreakerState.OPEN


def test_opens_after_3_consecutive_transient_failures_and_blocks(caplog):
    caplog.set_level(logging.INFO, logger="mad")
    b = CircuitBreaker("pm", failure_threshold=3, cooldown_s=30, clock=clock)
    for _ in range(3):
        with pytest.raises(TransientError):
            b.call(fail)
    assert b.state is BreakerState.OPEN
    called = []
    with pytest.raises(CircuitOpenError) as ei:
        b.call(lambda: called.append(1))
    assert ei.value.breaker == "pm"
    assert called == []
    assert b.failures == 3  # the rejection is not a failure
    assert [r for r in caplog.records
            if r.levelname == "WARNING" and "breaker_transition" in r.getMessage()]


def test_half_open_after_cooldown_probe_success_closes():
    b = _breaker()
    _open(b)
    t[0] = 30.0
    assert b.state is BreakerState.HALF_OPEN
    assert b.call(ok) == "ok"
    assert b.state is BreakerState.CLOSED
    assert b.failures == 0


def test_half_open_probe_failure_reopens():
    b = _breaker()
    _open(b)
    t[0] = 30.0
    with pytest.raises(TransientError):
        b.call(fail)
    assert b.state is BreakerState.OPEN
    t[0] = 59.9  # cool-down restarts from the failed probe
    with pytest.raises(CircuitOpenError):
        b.call(ok)


def test_half_open_allows_single_probe():
    b = _breaker()
    _open(b)
    t[0] = 30.0
    seen = []

    def probe():
        with pytest.raises(CircuitOpenError):
            b.call(ok)
        seen.append("rejected")
        return "probed"

    assert b.call(probe) == "probed"
    assert seen == ["rejected"]
    assert b.state is BreakerState.CLOSED


@pytest.mark.parametrize("exc", [PermanentError("p"), KeyboardInterrupt(), ValueError("v")])
def test_probe_slot_released_on_non_transient_exit(exc):
    b = _breaker()
    _open(b)
    t[0] = 30.0

    def boom():
        raise exc

    with pytest.raises(type(exc)):
        b.call(boom)
    assert b.state is BreakerState.HALF_OPEN
    assert b.call(ok) == "ok"  # slot free again, probe succeeds
    assert b.state is BreakerState.CLOSED


def test_success_resets_count():
    b = _breaker()
    for f in (fail, fail):
        with pytest.raises(TransientError):
            b.call(f)
    assert b.call(ok) == "ok"
    assert b.failures == 0
    for f in (fail, fail):
        with pytest.raises(TransientError):
            b.call(f)
    assert b.state is BreakerState.CLOSED
    assert b.failures == 2


def test_permanent_errors_neither_count_nor_reset():
    b = _breaker()
    for _ in range(2):
        with pytest.raises(TransientError):
            b.call(fail)
    for _ in range(5):
        with pytest.raises(PermanentError):
            b.call(_perm)
    assert b.failures == 2
    assert b.state is BreakerState.CLOSED
    with pytest.raises(TransientError):
        b.call(fail)
    assert b.state is BreakerState.OPEN
    assert b.failures == 3
    b2 = _breaker()
    for _ in range(10):
        with pytest.raises(PermanentError):
            b2.call(_perm)
    assert b2.state is BreakerState.CLOSED


def test_parsing_errors_do_not_count():
    b = _breaker()
    for _ in range(10):
        with pytest.raises(ValueError):
            b.call(_val)
    assert b.state is BreakerState.CLOSED
    assert b.failures == 0


def test_cooldown_configurable():
    b = _breaker(cooldown_s=5)
    _open(b)
    t[0] = 4.9
    assert b.state is BreakerState.OPEN
    t[0] = 5.0
    assert b.state is BreakerState.HALF_OPEN


def test_registry_shares_breaker_per_name():
    reg = BreakerRegistry()
    assert reg.get("pm") is reg.get("pm")
    assert reg.get("qa") is not reg.get("pm")
    reg.reset()
    assert reg.get("pm").failures == 0


def test_registry_configure_updates_existing_and_new():
    reg = BreakerRegistry()
    old = reg.get("pm")
    reg.configure(failure_threshold=1, cooldown_s=5, clock=clock)
    new = reg.get("qa")
    for b in (old, new):
        with pytest.raises(TransientError):
            b.call(fail)
        assert b.state is BreakerState.OPEN
    t[0] = 5.0
    assert old.state is BreakerState.HALF_OPEN


def test_runtime_resilient_call_retries_then_trips_breaker():
    runtime.reset()
    calls = []

    def flaky():
        calls.append(1)
        raise TransientError("x")

    with pytest.raises(TransientError):
        runtime.resilient_call(flaky, breaker="pm", classify=classify_llm_error, name="n")
    assert len(calls) == 4  # original + 3 retries, zero delay
    assert runtime.runtime().breakers.get("pm").failures == 1
    assert runtime.resilient_call(ok, breaker="pm", classify=classify_llm_error, name="n") == ("ok", 1)


def test_runtime_configure_from_settings(settings):
    settings.retry_max = 1
    settings.breaker_failure_threshold = 2
    runtime.configure(settings)
    assert runtime.runtime().policy == RetryPolicy(1, settings.retry_base_delay_s, settings.retry_max_delay_s)
    b = runtime.runtime().breakers.get("coder")
    for _ in range(2):
        with pytest.raises(TransientError):
            b.call(fail)
    assert b.state is BreakerState.OPEN
    runtime.reset()
    assert runtime.runtime().policy.base_delay_s == 0
    assert runtime.runtime().breakers.get("coder").state is BreakerState.CLOSED


def _nested_open():
    raise CircuitOpenError("other open", breaker="other")


def test_nested_circuit_open_neither_counts_nor_resets_in_closed():
    b = _breaker()
    with pytest.raises(TransientError):
        b.call(fail)
    with pytest.raises(CircuitOpenError) as ei:
        b.call(_nested_open)
    assert ei.value.breaker == "other"
    assert b.failures == 1
    assert b.state is BreakerState.CLOSED


def test_nested_circuit_open_in_half_open_probe_releases_slot_unchanged():
    b = _breaker()
    _open(b)
    t[0] = 30.0
    with pytest.raises(CircuitOpenError):
        b.call(_nested_open)
    assert b.failures == 3
    assert b.state is BreakerState.HALF_OPEN
    assert b.call(ok) == "ok"  # slot was released
    assert b.state is BreakerState.CLOSED

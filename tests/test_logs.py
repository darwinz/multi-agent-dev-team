"""JSON event logs, run/agent context and secret redaction."""

import io
import json
import logging
import sys

from observability.context import RunContext, agent_scope, run_scope
from observability.logs import configure_logging, log_event, redact, run_log_file
from resilience.errors import classify_llm_error
from resilience.retry import RetryPolicy, call_with_retry


class Flaky(Exception):
    status_code = 429


def test_event_is_one_json_line_with_context(tmp_path):
    ctx = RunContext(run_id="abc123")
    with run_scope(ctx), agent_scope("qa"), run_log_file(tmp_path / "events.jsonl"):
        log_event("cache_hit", key="k1")
    rec = json.loads((tmp_path / "events.jsonl").read_text().strip())
    assert rec["event"] == "cache_hit" and rec["run_id"] == "abc123" and rec["agent"] == "qa"
    assert rec["level"] == "INFO" and rec["key"] == "k1" and "ts" in rec


def test_secret_values_redacted(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-supersecret-123")
    with run_log_file(tmp_path / "e.jsonl"):
        log_event("llm_error", message="401 invalid key sk-ant-supersecret-123")
    assert "supersecret" not in (tmp_path / "e.jsonl").read_text()
    assert redact("x sk-ant-supersecret-123 y") == "x [redacted] y"


def test_file_handler_removed_after_scope(tmp_path):
    with run_log_file(tmp_path / "a.jsonl"):
        pass
    log_event("after")
    assert (tmp_path / "a.jsonl").read_text() == ""


def test_file_handler_removed_on_exception(tmp_path):
    logger = logging.getLogger("mad")
    before = list(logger.handlers)
    try:
        with run_log_file(tmp_path / "x.jsonl"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert logger.handlers == before


def test_retry_logs_attempts(caplog):
    calls = {"n": 0}
    def fn():
        calls["n"] += 1
        if calls["n"] <= 2:
            raise Flaky("busy")
        return "ok"
    P = RetryPolicy(max_retries=3, base_delay_s=0, max_delay_s=0)
    with caplog.at_level(logging.INFO, logger="mad"):
        call_with_retry(fn, policy=P, classify=classify_llm_error, name="t", sleep=lambda s: None)
    recs = [r for r in caplog.records if r.getMessage() == "retry_attempt"]
    assert [r.fields["attempt"] for r in recs] == [1, 2]


def test_non_string_fields_redacted(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-supersecret-123")
    secret = "sk-ant-supersecret-123"
    with run_log_file(tmp_path / "e.jsonl"):
        log_event("llm_error", error=RuntimeError(f"401 {secret}"), args=[secret],
                  body={"detail": secret})
    text = (tmp_path / "e.jsonl").read_text()
    assert "supersecret" not in text and "[redacted]" in text


def test_overlapping_run_logs_are_separate(tmp_path):
    a, b = RunContext(run_id="A"), RunContext(run_id="B")
    with run_scope(a), run_log_file(tmp_path / "a.jsonl"):
        with run_scope(b), run_log_file(tmp_path / "b.jsonl"):
            log_event("from_b")
        log_event("from_a")
    ta, tb = (tmp_path / "a.jsonl").read_text(), (tmp_path / "b.jsonl").read_text()
    assert "from_a" in ta and "from_b" not in ta
    assert "from_b" in tb and "from_a" not in tb


def test_scope_teardown_tolerates_other_context():
    import contextvars

    from observability.context import current_agent, current_run

    def gen():
        with run_scope(RunContext(run_id="g")), agent_scope("x"):
            yield

    g = gen()
    contextvars.copy_context().run(next, g)
    # closing from another Context would raise ValueError without tolerant teardown
    contextvars.copy_context().run(g.close)
    assert current_run() is None and current_agent() is None


def test_stderr_handler_follows_replaced_stderr(monkeypatch):
    """configure_logging is called once per CLI run; a later sys.stderr (pytest capture) must not go stale."""
    first, second = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stderr", first)
    configure_logging("INFO")
    monkeypatch.setattr(sys, "stderr", second)
    first.close()  # the first stream is gone: writing to it would print a "Logging error" traceback
    configure_logging("INFO")
    log_event("probe_event")
    assert json.loads(second.getvalue().strip().splitlines()[-1])["event"] == "probe_event"

"""JSON event logs, run/agent context and secret redaction."""

import json
import logging

from observability.context import RunContext, agent_scope, run_scope
from observability.logs import log_event, redact, run_log_file
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

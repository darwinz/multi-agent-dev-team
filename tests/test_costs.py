import json

import pytest
from langchain_core.messages import AIMessage

from observability.costs import (
    PRICING,
    CallRecord,
    build_cost_report,
    cost_usd,
    usage_from_message,
    write_cost_report,
)


def rec(role, p, c, cost, hit=False):
    return CallRecord(ts="t", role=role, agent=role, model="anthropic:claude-haiku-5-5", prompt_tokens=p,
                      completion_tokens=c, cache_read_tokens=0, cache_write_tokens=0, cache_hit=hit,
                      cost_usd=cost, priced=True, duration_ms=1.0, attempts=1)


def test_cost_from_price_table():
    pr = PRICING["claude-haiku-5-5"]
    cost, priced = cost_usd("anthropic:claude-haiku-5-5", 1_000_000, 1_000_000, 0, 0)
    assert priced and cost == pytest.approx(pr.input + pr.output)


def test_cache_tokens_priced_separately():
    pr = PRICING["claude-sonnet-5-5"]
    cost, _ = cost_usd("anthropic:claude-sonnet-5-5", 1_000_000, 0, 400_000, 100_000)
    assert cost == pytest.approx(0.5 * pr.input + 0.4 * pr.cache_read + 0.1 * pr.cache_write)


def test_unknown_model_costs_zero_and_flags(caplog):
    with caplog.at_level("WARNING", logger="mad"):
        assert cost_usd("ollama:llama3", 10, 10, 0, 0) == (0.0, False)
    assert any(r.getMessage() == "unpriced_model" and r.levelname == "WARNING" for r in caplog.records)


def test_usage_from_message():
    m = AIMessage("x", usage_metadata={"input_tokens": 120, "output_tokens": 30, "total_tokens": 150,
                                       "input_token_details": {"cache_read": 100, "cache_creation": 0}})
    assert usage_from_message(m) == (120, 30, 100, 0)


def test_usage_from_message_absent():
    assert usage_from_message(AIMessage("x")) == (0, 0, 0, 0)


def test_report_totals_reconcile_with_calls():
    calls = [rec("pm", 100, 50, 0.01), rec("coder", 200, 80, 0.02), rec("qa", 10, 5, 0.0, hit=True)]
    r = build_cost_report(run_id="r1", started_at="a", finished_at="b", requirement="q", outcome="success",
                          models={"pm": "m"}, calls=calls, trace_url=None)
    assert r["total"]["prompt_tokens"] == sum(c.prompt_tokens for c in calls) == 310
    assert r["total"]["cost_usd"] == pytest.approx(sum(a["cost_usd"] for a in r["agents"].values()))
    assert r["agents"]["qa"]["cache_hits"] == 1 and len(r["calls"]) == 3


def test_report_empty_agents_pricing_and_unpriced():
    r = build_cost_report(run_id="r1", started_at="a", finished_at="b", requirement="q", outcome="failed",
                          models={}, calls=[], trace_url=None, error={"type": "X"})
    assert set(r["agents"]) == {"pm", "coder", "qa"} and r["agents"]["pm"]["calls"] == 0
    assert r["unpriced_models"] == [] and r["error"] == {"type": "X"}
    odd = rec("pm", 1, 1, 0.0)
    odd.model, odd.priced = "ollama:llama3", False
    r = build_cost_report(run_id="r1", started_at="a", finished_at="b", requirement="q", outcome="success",
                          models={}, calls=[odd, rec("qa", 1, 1, 0.0)], trace_url=None)
    assert r["unpriced_models"] == ["ollama:llama3"]
    assert set(r["pricing"]) == {"as_of", "claude-haiku-5-5"}


def test_write_cost_report_two_copies(tmp_path):
    r = build_cost_report(run_id="r1", started_at="a", finished_at="b", requirement="q", outcome="success",
                          models={}, calls=[rec("pm", 1, 1, 0.01)], trace_url=None)
    reports, run_dir = tmp_path / "docs" / "cost_reports", tmp_path / ".coder" / "runs" / "r1"
    written, errors = write_cost_report(r, reports, run_dir)
    copy = run_dir / "cost-report.json"
    assert written == [copy, reports / "r1.json"] and errors == []  # run-dir copy first
    assert written[1].read_text() == copy.read_text()
    assert json.loads(written[1].read_text())["run_id"] == "r1"


def test_write_cost_report_copies_are_independent(tmp_path):
    r = build_cost_report(run_id="r1", started_at="a", finished_at="b", requirement="q", outcome="success",
                          models={}, calls=[], trace_url=None)
    (tmp_path / "docs").write_text("a file, so docs/cost_reports can't be created")
    run_dir = tmp_path / ".coder" / "runs" / "r1"
    written, errors = write_cost_report(r, tmp_path / "docs" / "cost_reports", run_dir)
    assert written == [run_dir / "cost-report.json"] and len(errors) == 1 and isinstance(errors[0], OSError)

    blocked = tmp_path / "blocked"
    blocked.write_text("a file, so the run dir can't be created")
    written, errors = write_cost_report(r, tmp_path / "reports", blocked / "r1")
    assert written == [tmp_path / "reports" / "r1.json"] and len(errors) == 1  # docs copy still attempted


def test_usage_counts_cache_writes_reported_by_ttl():
    # langchain-anthropic zeroes cache_creation and reports writes per TTL when the API breaks them down
    m = AIMessage("x", usage_metadata={
        "input_tokens": 5000, "output_tokens": 30, "total_tokens": 5030,
        "input_token_details": {"cache_read": 3000, "cache_creation": 0,
                                "ephemeral_5m_input_tokens": 1500, "ephemeral_1h_input_tokens": 200}})
    assert usage_from_message(m) == (5000, 30, 3000, 1700)

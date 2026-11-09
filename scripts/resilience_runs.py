"""Generate the v4 resilience evidence: `uv run python -m scripts.resilience_runs`.

Scripted models only, no keys. One directory per scenario holds `events.jsonl` (the captured JSON log) and
`summary.md`; the temporary project root is replaced by `<project>`. Each scenario asserts what it shows,
so main() returns 0 only when the evidence holds.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage  # noqa: E402

from agents.config import Settings  # noqa: E402
from agents.model import resilient  # noqa: E402
from observability.context import RunContext, run_scope  # noqa: E402
from observability.logs import log_event, run_log_file  # noqa: E402
from orchestration.project_state import Spec  # noqa: E402
from resilience import runtime as resilience_runtime  # noqa: E402
from resilience.errors import CircuitOpenError, TransientError  # noqa: E402
from scripts.controlled_runs import _scrub  # noqa: E402
from tests.fakes import ScriptedChatModel, ai, make_spec, qa_suite_response, spec_response  # noqa: E402
from tests.scenarios import STACK_TESTS, passing_scripts, scripted_runner  # noqa: E402

DEFAULT_OUT = Path("docs/evidence/v4/controlled")
LABEL = "scripted models, no API keys"
REQUIREMENT = "Build a linked list"
COOLDOWN_S = 0.2
CACHE_TTL_S = 2.0


class RateLimited(Exception):
    """What a provider's 429 looks like to the classifier."""

    status_code = 429


def _settings(project: Path, **overrides) -> Settings:
    base = dict(project_dir=project, model="scripted", exec_timeout_s=15, retry_base_delay_s=0.05,
                retry_max_delay_s=0.2, cache_ttl_s=0.0)
    return Settings.from_env(**{**base, **overrides})


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _named(events: list[dict], name: str) -> list[dict]:
    return [e for e in events if e["event"] == name]


def _pm_call(model, text: str = REQUIREMENT):
    return model.cached().with_structured_output(Spec, include_raw=True).invoke([HumanMessage(text)])


# Each scenario takes (project dir, scenario dir) and returns (what it shows, extra summary lines).
# It writes events.jsonl (and any other files) into the scenario dir and asserts its own claims.

def retry_then_success(project: Path, dest: Path) -> tuple[str, list[str]]:
    settings = _settings(project)
    with run_scope(RunContext(run_id="retry-then-success")), run_log_file(dest / "events.jsonl"):
        m = resilient(ScriptedChatModel(responses=[RateLimited("429"), ai("hello")]), settings, "pm")
        assert m.invoke([HumanMessage("x")]).content == "hello"
    events = _events(dest / "events.jsonl")
    assert [e["attempt"] for e in _named(events, "retry_attempt")] == [1]
    assert not _named(events, "retry_exhausted")
    assert [e["attempts"] for e in _named(events, "llm_call")] == [2]
    return ("The PM model answers one HTTP 429, then succeeds. The wrapper logs one `retry_attempt` (WARNING, "
            "with the jittered delay) and the `llm_call` records `attempts: 2`.",
            ["retry_attempt events: 1", "llm_call attempts: 2"])


def retry_exhausted(project: Path, dest: Path) -> tuple[str, list[str]]:
    settings = _settings(project)
    runner = scripted_runner(settings, ([RateLimited("429")] * 4, [], []))
    state = runner.run(REQUIREMENT)
    assert runner.run_dir is not None
    events = _events(runner.run_dir / "events.jsonl")
    assert [e["attempt"] for e in _named(events, "retry_attempt")] == [1, 2, 3]
    assert [e["attempt"] for e in _named(events, "retry_exhausted")] == [4]
    assert state["status"] == "failed"
    report = json.loads((runner.run_dir / "cost-report.json").read_text(encoding="utf-8"))
    assert report["outcome"] == "failed"
    (dest / "events.jsonl").write_text((runner.run_dir / "events.jsonl").read_text(encoding="utf-8"),
                                       encoding="utf-8")
    (dest / "cost-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return ("The PM model answers four 429s in a row: the original attempt and the three allowed retries. "
            "The wrapper logs three `retry_attempt` events and one `retry_exhausted` (attempt 4), the PM "
            "fails cleanly and the run is `failed`. `cost-report.json` is the run's report: outcome "
            f"`failed`, no billable calls. Run error: `{state['error']}`.",
            ["retry_attempt events: 3", "retry_exhausted events: 1 (attempt 4)", "run status: failed",
             f"cost report outcome: {report['outcome']}"])


def breaker(project: Path, dest: Path) -> tuple[str, list[str]]:
    """Leg 1 uses the default retry_max=3 (retry composed with the breaker); leg 2 uses retry_max=0 only to
    keep the half-open cycles short: with no retries each call is a single attempt."""
    threshold = 3
    with run_scope(RunContext(run_id="breaker")), run_log_file(dest / "events.jsonl"):
        # Leg 1: default retries. Each call = 4 attempts (all 429) -> retry_exhausted -> one breaker failure.
        settings = _settings(project / "leg1", breaker_failure_threshold=threshold, breaker_cooldown_s=COOLDOWN_S)
        assert settings.retry_max == 3
        log_event("scenario_step", step="leg 1: default retry_max=3, three calls that each exhaust their retries")
        inner = ScriptedChatModel(responses=[RateLimited("429")] * 12)
        m = resilient(inner, settings, "pm")
        for i in range(1, 4):
            try:
                m.invoke([HumanMessage(f"leg 1 call {i}")])
            except TransientError:
                log_event("scenario_step", step=f"leg 1 call {i}", outcome="transient_failure",
                          inner_model_calls=len(inner.calls), breaker_failures=resilience_runtime.runtime()
                          .breakers.get("pm").failures)
        calls_when_open = len(inner.calls)
        try:
            m.invoke([HumanMessage("leg 1 call 4")])
            raise AssertionError("the open breaker let a call through")
        except CircuitOpenError:
            log_event("call_blocked", step="leg 1 call 4", inner_model_calls=len(inner.calls),
                      note="breaker open: the model was not called")
        assert calls_when_open == 12 and len(inner.calls) == 12  # 3 calls x 4 attempts; the blocked call adds none

        # Leg 2: retry_max=0, two full open -> half_open -> closed/open cycles
        resilience_runtime.reset()
        settings = _settings(project / "leg2", retry_max=0, breaker_failure_threshold=threshold,
                             breaker_cooldown_s=COOLDOWN_S)
        log_event("scenario_step", step="leg 2: retry_max=0 (one attempt per call) for the half-open cycles")
        script = ([RateLimited("429")] * 3 + [ai("probe ok")] + [RateLimited("429")] * 3
                  + [RateLimited("probe fails")])
        m = resilient(ScriptedChatModel(responses=script), settings, "pm")

        def attempt(label: str) -> None:
            try:
                m.invoke([HumanMessage(label)])
                log_event("scenario_step", step=label, outcome="ok")
            except CircuitOpenError:
                log_event("call_blocked", step=label, note="breaker open: the model was not called")
            except TransientError:
                log_event("scenario_step", step=label, outcome="transient_failure")

        for i in range(1, 4):
            attempt(f"cycle 1 failure {i}")
        attempt("cycle 1 call while open")
        time.sleep(COOLDOWN_S + 0.05)
        attempt("cycle 1 probe (succeeds)")
        for i in range(1, 4):
            attempt(f"cycle 2 failure {i}")
        time.sleep(COOLDOWN_S + 0.05)
        attempt("cycle 2 probe (fails)")
    events = _events(dest / "events.jsonl")
    leg1_end = next(i for i, e in enumerate(events) if e["event"] == "scenario_step" and str(e["step"]).startswith("leg 2"))
    leg1, leg2 = events[:leg1_end], events[leg1_end:]
    assert [e["attempt"] for e in _named(leg1, "retry_attempt")] == [1, 2, 3] * 3
    assert [e["attempt"] for e in _named(leg1, "retry_exhausted")] == [4, 4, 4]
    assert [(e["from_state"], e["to_state"], e["failures"]) for e in _named(leg1, "breaker_transition")] == [
        ("closed", "open", 3)]
    assert [e["breaker_failures"] for e in leg1 if e.get("outcome") == "transient_failure"] == [1, 2, 3]
    transitions = _named(leg2, "breaker_transition")
    seq = [(e["from_state"], e["to_state"]) for e in transitions]
    assert seq == [("closed", "open"), ("open", "half_open"), ("half_open", "closed"),
                   ("closed", "open"), ("open", "half_open"), ("half_open", "open")], seq
    assert all(e["level"] == "WARNING" for e in _named(events, "breaker_transition"))
    assert len(_named(leg2, "call_blocked")) == 1
    return (f"The `pm` breaker with threshold {threshold} and a real {COOLDOWN_S} s cool-down, in two legs "
            "(`scenario_step` and `call_blocked` lines are harness annotations, not production events).\n\n"
            "**Leg 1, retry composed with the breaker (default `retry_max=3`).** Three calls each make 4 "
            "attempts (12 model calls in all, 3 `retry_attempt` then `retry_exhausted` at attempt 4 per call). "
            "Each exhausted call is one breaker failure (1, 2, 3), so the third opens the breaker "
            "(closed->open). A fourth call is blocked without reaching the model: the model call count "
            "stays at 12.\n\n"
            "**Leg 2, half-open cycles (`retry_max=0`, only to keep the run short: one attempt per call).** "
            "Cycle 1: three failures open it, a call while open is blocked, after the cool-down the probe "
            "succeeds and it closes. Cycle 2: three more failures open it, and after the cool-down the probe "
            "fails and it reopens. Every `breaker_transition` is a WARNING.",
            ["leg 1 retry_attempt events: 9, retry_exhausted events: 3 (attempt 4 each)",
             "leg 1 model calls: 12 before the breaker opened, 12 after the blocked call",
             "leg 1 breaker failures after each call: 1, 2, 3 -> closed->open",
             "leg 2 transitions: " + ", ".join(f"{a}->{b}" for a, b in seq),
             "all transitions WARNING: yes"])


def cache(project: Path, dest: Path) -> tuple[str, list[str]]:
    settings = _settings(project, cache_ttl_s=CACHE_TTL_S)
    spec = make_spec()
    with run_scope(RunContext(run_id="cache")), run_log_file(dest / "events.jsonl"):
        inner = ScriptedChatModel(responses=[spec_response(spec), spec_response(spec)])
        m = resilient(inner, settings, "pm")
        _pm_call(m)
        _pm_call(m)  # same request inside the TTL: served from disk
        calls_before_expiry = len(inner.calls)
        time.sleep(CACHE_TTL_S + 0.1)
        _pm_call(m)  # past the TTL: the entry is stale, so the model is asked again
    events = _events(dest / "events.jsonl")
    names = [e["event"] for e in events if e["event"].startswith("cache_")]
    assert names == ["cache_miss", "cache_hit", "cache_expired"], names
    assert calls_before_expiry == 1 and len(inner.calls) == 2
    hits = [e["cache_hit"] for e in _named(events, "llm_call")]
    assert hits == [False, True, False], hits
    return (f"A PM spec request with a {CACHE_TTL_S:g} s response-cache TTL. First call: `cache_miss`, the model "
            "answers. Same request again: `cache_hit` (zero-cost `llm_call`, `attempts: 0`, the model is not "
            f"called). After sleeping {CACHE_TTL_S + 0.1:g} s: `cache_expired`, the model is called again.",
            ["cache events: " + ", ".join(names), f"model calls: {len(inner.calls)} for 3 requests"])


def invalid_output(project: Path, dest: Path) -> tuple[str, list[str]]:
    markers = dest / "markers.tmp"
    with run_scope(RunContext(run_id="invalid-output")), run_log_file(markers):
        # PM probe: two replies that are not a Spec
        log_event("scenario_step", step="pm-probe")
    pm_runner = scripted_runner(_settings(project / "pm"), ([ai("not a spec"), ai("still not")], [], []))
    pm_state = pm_runner.run(REQUIREMENT)
    assert pm_state["status"] == "failed" and "invalid spec twice" in pm_state["error"]
    with run_scope(RunContext(run_id="invalid-output")), run_log_file(dest / "markers2.tmp"):
        # QA probe: the Coder delivers, then QA twice returns a suite whose manifest names a test not in the code
        log_event("scenario_step", step="qa-probe")
    pm, coder, _ = passing_scripts()
    bad = qa_suite_response(STACK_TESTS, (("test_nope", 1),))
    qa_runner = scripted_runner(_settings(project / "qa"), (pm, coder, [bad, bad]))
    qa_state = qa_runner.run(REQUIREMENT)
    outcomes = [t["qa_outcome"] for t in qa_state["task_status"].values()]
    assert outcomes == ["qa_error"], outcomes
    assert pm_runner.run_dir is not None and qa_runner.run_dir is not None
    pm_events = _events(pm_runner.run_dir / "events.jsonl")
    qa_events = _events(qa_runner.run_dir / "events.jsonl")
    # PM probe: nothing reached a tool
    assert not _named(pm_events, "tool_call")
    # QA probe: once QA's first model call is logged, no tool runs again (QA never wrote or executed a suite)
    first_qa = next(i for i, e in enumerate(qa_events) if e["event"] == "llm_call" and e.get("role") == "qa")
    assert not [e for e in qa_events[first_qa:] if e["event"] == "tool_call"]
    coder_tools = [e["tool"] for e in qa_events[:first_qa] if e["event"] == "tool_call"]
    assert coder_tools and all(e.get("agent") == "coder" for e in qa_events[:first_qa] if e["event"] == "tool_call")
    assert len([e for e in qa_events[first_qa:] if e["event"] == "llm_call" and e.get("role") == "qa"]) == 2
    # What production logs for the rejection: the QA report carries qa_error (no dedicated WARNING event exists)
    qa_report = json.loads(next(qa_runner.run_dir.glob("qa-T1-r*.json")).read_text(encoding="utf-8"))
    assert qa_report["qa_error"] and qa_report["approved"] is False
    warnings = [e["event"] for e in qa_events + pm_events if e["level"] in ("WARNING", "ERROR")]
    with (dest / "events.jsonl").open("w", encoding="utf-8") as f:
        for part in (markers, *pm_events, dest / "markers2.tmp", *qa_events):
            if isinstance(part, Path):
                f.write(part.read_text(encoding="utf-8"))
            else:
                f.write(json.dumps(part) + "\n")
    markers.unlink()
    (dest / "markers2.tmp").unlink()
    (dest / "qa-report.json").write_text(json.dumps(qa_report, indent=2), encoding="utf-8")
    return ("Two probes whose model returns invalid structured output (`scenario_step` lines are harness "
            "markers; each probe's section of `events.jsonl` follows its marker).\n\n"
            f"**pm-probe.** The PM returns two replies that are not a `Spec`. The run is `{pm_state['status']}` "
            f"(`{pm_state['error']}`) and the section has zero `tool_call` events.\n\n"
            "**qa-probe.** The Coder delivers normally (its own `write_file`/`exec_python` `tool_call` events "
            "come before QA's first model call and are legitimate). QA then returns two suites whose manifest "
            "names a test absent from the code. After QA's first `llm_call` there is no `tool_call` event: "
            f"nothing was written or executed for QA. The task ends `{outcomes[0]}`.\n\n"
            "**What is logged for the rejection.** Production emits no dedicated WARNING event for an invalid "
            "PM spec or QA suite. The evidence of the rejection is the run outcome (`failed` for the PM, "
            "`qa_error` for the task) and `qa-report.json` (`qa_error`: "
            f"`{qa_report['qa_error']}`).\n\n"
            "**Not claimed here.** The invalid *tool-call* probe (a `write_file` call with a missing or "
            "escaping argument rejected before any write) is covered by "
            "`tests/test_security.py::test_invalid_tool_call_rejected_before_write`, not by this run.",
            ["pm-probe: run status failed, tool_call events: 0",
             f"qa-probe: task outcome {outcomes[0]}, tool_call events after QA's first llm_call: 0",
             f"qa-probe: Coder tool_call events before QA (legitimate): {len(coder_tools)}",
             f"WARNING/ERROR events in either run: {len(warnings)}"])


SCENARIOS: dict[str, Callable[[Path, Path], tuple[str, list[str]]]] = {
    "retry-then-success": retry_then_success,
    "retry-exhausted": retry_exhausted,
    "breaker": breaker,
    "cache": cache,
    "invalid-output": invalid_output,
}


def _scrub_dir(dest: Path, roots: list[str]) -> None:
    for f in dest.rglob("*"):
        if f.is_file():
            f.write_text(_scrub(f.read_text(encoding="utf-8"), roots), encoding="utf-8")


def run_scenario(name: str, out: Path) -> bool:
    dest = out / name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    resilience_runtime.reset()
    with tempfile.TemporaryDirectory(prefix="resilience-") as tmp:
        project = Path(tmp)
        try:
            shows, facts = SCENARIOS[name](project, dest)
            passed, note = True, ""
        except AssertionError as e:
            shows, facts, passed, note = "The scenario's assertions did not hold.", [], False, f"\n\nFAILED: {e}"
        body = (f"# Resilience run: {name}\n\n{LABEL}.\n\nCommand: `uv run python -m scripts.resilience_runs`\n"
                f"Result: {'PASS' if passed else 'FAIL'}\n\n{shows}{note}\n\n"
                + "".join(f"- {line}\n" for line in facts)
                + "\nFiles: `events.jsonl` is the captured JSON log (one object per line); "
                  "`<project>` replaces the temporary project root.\n")
        (dest / "summary.md").write_text(body, encoding="utf-8")
        _scrub_dir(dest, [str(project), str(project.resolve())])
    resilience_runtime.reset()
    return passed


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="resilience_runs", description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output directory (default {DEFAULT_OUT})")
    args = p.parse_args(argv)
    results = {name: run_scenario(name, args.out) for name in SCENARIOS}
    for name, passed in results.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

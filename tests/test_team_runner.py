"""TeamRunner: one RunLock per project, evidence files under .coder/runs/<run_id>/."""

from __future__ import annotations

import json

import pytest

from agents.config import Settings
from agents.pm_agent import PMAgent
from agents.qa_agent import canonical_suite_path
from orchestration.a2a import CODER_ACCEPTS, QA_ACCEPTS
from orchestration.project_state import TaskHandoff
from orchestration.team_runner import TeamRunner
from tests.fakes import (
    PAL_REVISED,
    QA_PAL_TESTS,
    ScriptedChatModel,
    agent,
    ai,
    make_spec,
    qa_agent,
    spec_response,
    structured,
    task_scripts,
    tool_call,
)
from tests.scenarios import nonconverging_scripts, scripted_runner
from tools.workspace import RunLock, WorkspaceBusyError

REQ = "Build a small demo API"


@pytest.fixture
def settings(tmp_path) -> Settings:
    # These tests run real QA pytest through the team graph; 15 s keeps them stable under load
    return Settings.from_env(project_dir=tmp_path, exec_timeout_s=15)


def two_tasks() -> tuple[list, list, list]:
    (c1, q1), (c2, q2) = task_scripts(), task_scripts()
    return [spec_response(make_spec())], c1 + c2, q1 + q2


def one_task() -> tuple[list, list, list]:
    coder, qa = task_scripts()
    return [spec_response(make_spec(1))], coder, qa


def happy_runner(settings) -> TeamRunner:
    return scripted_runner(settings, two_tasks())


def bad_spec_responses() -> list:
    bad = make_spec().model_dump()
    for task in bad["tasks"]:
        task["covers"] = ["AC1"]
    return [structured("Spec", bad), structured("Spec", bad)]


def finish_only() -> list:
    return [ai("done", tool_call("finish", explanation="nothing", plan=["finish"], entry_point="x.py"))]


def load(runner: TeamRunner, name: str) -> dict:
    return json.loads((runner.run_dir / name).read_text())


def a2a_log(runner: TeamRunner) -> list[dict]:
    return [json.loads(line) for line in (runner.run_dir / "a2a-log.jsonl").read_text().splitlines()]


def test_evidence_files_written(settings):
    runner = happy_runner(settings)
    runner.run(REQ)
    assert runner.run_dir.parent == settings.state_dir / "runs"
    per_task = [f"{name}-{t}{ext}" for t in ("T1", "T2") for name, ext in
                [("handoff", ".json"), ("reflection", ".json"), ("qa", "-r1.json"), ("qa-tests", ".py"),
                 ("state-after", ".json")]]
    assert sorted(p.name for p in runner.run_dir.iterdir()) == sorted([
        "requirement.txt", "state-after-pm.json", *per_task, "a2a-log.jsonl", "regression.json",
        "final-state.json",
    ])
    assert (runner.run_dir / "requirement.txt").read_text() == REQ
    assert load(runner, "final-state.json") == runner.final_state
    assert runner.final_state["status"] == "completed"


def test_state_after_pm_has_tasks_but_no_progress(settings):
    runner = happy_runner(settings)
    runner.run(REQ)
    state = load(runner, "state-after-pm.json")
    assert state["tasks"]
    assert state["task_status"]
    assert all(r["status"] == "pending" for r in state["task_status"].values())
    assert state["code_output"] == {}


def test_handoff_file_matches_task_record(settings):
    runner = happy_runner(settings)
    runner.run(REQ)
    raw = (runner.run_dir / "handoff-T1.json").read_text()
    handoff = TaskHandoff.model_validate_json(raw)
    assert handoff.task_id == "T1"
    assert raw == handoff.model_dump_json(indent=2)


def test_state_after_task_written_at_close(settings):
    runner = happy_runner(settings)
    runner.run(REQ)
    before = load(runner, "state-after-pm.json")
    after = load(runner, "state-after-T1.json")
    assert before["task_status"]["T1"]["status"] == "pending"
    t1 = after["task_status"]["T1"]
    assert t1["status"] == "done" and t1["qa_outcome"] == "approved" and t1["qa_rounds"] == 1 and t1["reflected"]
    assert after["task_status"]["T2"]["status"] == "pending"
    assert after["handoff"] is None and after["coder_step"] == "close"
    for path in t1["files"]:
        assert path in after["code_output"]
    assert after["code_output"]["palindrome.py"] == PAL_REVISED
    assert load(runner, "state-after-T2.json")["task_status"]["T2"]["status"] == "done"


def test_a2a_log_written(settings):
    runner = scripted_runner(settings, one_task())
    runner.run(REQ)
    lines = a2a_log(runner)
    assert lines[0] == {"event": "capabilities", "agent": "coder",
                        "accepts": [{"sender": s, "intent": i} for s, i in sorted(CODER_ACCEPTS)]}
    assert lines[1] == {"event": "capabilities", "agent": "qa",
                        "accepts": [{"sender": s, "intent": i} for s, i in sorted(QA_ACCEPTS)]}
    records = lines[2:]
    assert [(r["event"], r["message"]["intent"]) for r in records] == [
        ("sent", "review_request"), ("received", "review_request"), ("sent", "approved"), ("received", "approved")]
    for r in records:
        assert set(r["message"]) == {"sender", "receiver", "intent", "payload", "correlation_id"}
        assert r["reason"] is None and r["at"]
    assert len({r["message"]["correlation_id"] for r in records}) == 1
    assert records[0]["message"]["sender"] == "coder" and records[2]["message"]["sender"] == "qa"


def test_reflection_and_qa_evidence_files(settings):
    runner = scripted_runner(settings, one_task())
    runner.run(REQ)
    reflection = load(runner, "reflection-T1.json")
    assert reflection["task_id"] == "T1" and reflection["changed"] == ["palindrome.py"]
    assert reflection["critique"]["items"] and reflection["error"] is None and reflection["dropped"] == []
    (request,) = [r for r in a2a_log(runner)[2:] if r["event"] == "sent" and r["message"]["intent"] == "review_request"]
    assert request["message"]["payload"]["files"] == reflection["after"]

    report = load(runner, "qa-T1-r1.json")
    assert report == runner.final_state["qa_reports"][0]
    assert report["approved"] is True and report["tests_run"] == 1 and report["round"] == 1

    tests = (runner.run_dir / "qa-tests-T1.py").read_text()
    assert tests == QA_PAL_TESTS == canonical_suite_path(settings, "T1").read_text()

    regression = load(runner, "regression.json")
    assert regression == runner.final_state["regression"]
    assert regression["approved"] is True and set(regression["suites_run"]) == {"T1"}


def test_qa_max_iterations_from_settings(settings):
    s = settings.model_copy(update={"qa_max_iterations": 2})
    runner = scripted_runner(s, nonconverging_scripts(2))
    state = runner.run(REQ)
    assert state["status"] == "partial"
    rec = state["task_status"]["T1"]
    assert rec["status"] == "failed" and rec["qa_outcome"] == "unresolved" and rec["qa_rounds"] == 2
    assert rec["unresolved"] and "test_reverse" in rec["unresolved"][0]
    assert load(runner, "qa-T1-r1.json")["approved"] is False
    assert load(runner, "qa-T1-r2.json")["approved"] is False
    assert not (runner.run_dir / "qa-T1-r3.json").exists()
    assert not (runner.run_dir / "regression.json").exists()  # no task was done
    (final,) = [r for r in a2a_log(runner) if r.get("event") == "sent" and r["message"]["intent"] == "final_report"]
    assert final["message"]["payload"]["reason"] == "max_iterations"


def test_consolidation_file_when_over_8(settings):
    s = settings.model_copy(update={"max_iterations": 2})
    # Each merged task: an empty delivery, a critique that fails twice (no revision), QA generation fails
    coder = [r for _ in range(8) for r in (*finish_only(), RuntimeError("no critique"), RuntimeError("again"))]
    pm = [spec_response(make_spec(11)), RuntimeError("no merge")]
    runner = scripted_runner(s, (pm, coder, [RuntimeError("no suite")] * 8))
    events = list(runner.stream(REQ))
    consolidation = load(runner, "consolidation.json")
    flat = sorted(i for ids in consolidation["merged_from"].values() for i in ids)
    assert flat == sorted(f"T{i}" for i in range(1, 12))
    assert consolidation["method"] == "deterministic"
    kinds = [e.kind for e in events]
    assert kinds.index("spec") < kinds.index("consolidation") < kinds.index("handoff")


def test_event_order(settings):
    runner = happy_runner(settings)
    events = list(runner.stream(REQ))
    kinds = [e.kind for e in events]
    assert kinds[:3] == ["start", "spec", "handoff"]
    assert kinds[-1] == "final"
    assert events[0].data == {"run_id": runner.run_dir.name, "run_dir": str(runner.run_dir)}
    for task_id in ("T1", "T2"):
        h = next(i for i, e in enumerate(events) if e.kind == "handoff" and e.task_id == task_id)
        (d,) = [i for i, e in enumerate(events) if e.kind == "task_done" and e.task_id == task_id]
        between = events[h + 1:d]
        assert {e.task_id for e in between} == {task_id}
        order = [e.kind for e in between if e.kind != "coder"]
        assert order.index("reflection") < order.index("a2a") < order.index("qa_report")
        assert events[d].data["status"] == "done"
    assert events[-1].data["state"] == runner.final_state
    assert events[-1].data["api_check"] is None and events[-1].data["demo_check"] is None


def test_busy_lock_no_run_dir(settings):
    runner = happy_runner(settings)
    with RunLock(settings.state_dir):
        with pytest.raises(WorkspaceBusyError):
            list(runner.stream("r"))
    runs = settings.state_dir / "runs"
    assert not runs.exists() or not any(runs.iterdir())
    assert runner.run_dir is None


def test_lock_held_across_tasks(settings):
    runner = happy_runner(settings)
    seen = []
    for event in runner.stream(REQ):
        if event.kind == "task_done" and not seen:
            seen.append(RunLock.is_running(settings.state_dir))
    assert seen == [True]
    assert not RunLock.is_running(settings.state_dir)


def test_api_check_runs_under_lock(settings):
    runner = happy_runner(settings)
    calls = []

    def api_check(workspace, written):
        calls.append((workspace, written))
        assert RunLock(settings.state_dir).is_locked_elsewhere()
        with pytest.raises(WorkspaceBusyError):
            list(TeamRunner(settings, pm=runner.pm, coder=runner.coder, qa=runner.qa).stream("r"))
        return False, "FAIL echo"

    events = list(runner.stream(REQ, api_check=api_check))
    assert calls == [(settings.workspace_dir, ["palindrome.py", "test_palindrome.py"])]
    assert "FAIL echo" in (runner.run_dir / "api-check.txt").read_text()
    assert events[-1].data["api_check"] == {"passed": False, "report": "FAIL echo"}
    assert [p.name for p in (settings.state_dir / "runs").iterdir()] == [runner.run_dir.name]


def test_demo_check_runs_under_lock(settings):
    runner = scripted_runner(settings, one_task())
    calls = []

    def demo_check(workspace, state):
        calls.append((workspace, state["status"]))
        assert RunLock(settings.state_dir).is_locked_elsewhere()
        assert state["task_status"]["T1"]["status"] == "done"
        return True, "PASS everything"

    events = list(runner.stream(REQ, demo_check=demo_check))
    assert calls == [(settings.workspace_dir, "completed")]
    assert (runner.run_dir / "demo-check.txt").read_text() == "PASS everything"
    assert events[-1].data["demo_check"] == {"passed": True, "report": "PASS everything"}
    assert events[-1].data["api_check"] is None


def test_api_check_skipped_when_pm_failed(settings):
    runner = scripted_runner(settings, (bad_spec_responses(), [], []))
    calls = []

    def check(workspace, arg):
        calls.append(workspace)
        return True, "PASS"

    events = list(runner.stream(REQ, api_check=check, demo_check=check))
    assert calls == []
    assert not (runner.run_dir / "api-check.txt").exists()
    assert not (runner.run_dir / "demo-check.txt").exists()
    assert "error" in next(e for e in events if e.kind == "spec").data
    assert events[-1].data["state"]["status"] == "failed"
    assert events[-1].data["api_check"] is None and events[-1].data["demo_check"] is None
    assert [line["event"] for line in a2a_log(runner)] == ["capabilities", "capabilities"]


def test_pm_setup_failure_writes_failed_state(settings):
    class NoStructured(ScriptedChatModel):
        def with_structured_output(self, *a, **kw):
            raise NotImplementedError("structured output not supported")

    runner = TeamRunner(settings, pm=PMAgent(settings, model=NoStructured(responses=[])), coder=agent(settings),
                        qa=qa_agent(settings))
    state = runner.run("Build a REST API")
    assert state["status"] == "failed"
    assert "not supported" in state["error"]
    assert load(runner, "final-state.json")["status"] == "failed"

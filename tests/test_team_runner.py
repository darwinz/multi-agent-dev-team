"""TeamRunner: one RunLock per project, evidence files under .coder/runs/<run_id>/."""

from __future__ import annotations

import json

import pytest

from agents.pm_agent import PMAgent
from orchestration.project_state import TaskHandoff
from orchestration.team_runner import TeamRunner
from tests.fakes import ScriptedChatModel, agent, ai, happy, make_spec, spec_response, structured, tool_call
from tools.workspace import RunLock, WorkspaceBusyError

REQ = "Build a small demo API"


def make_runner(settings, pm_responses, *coder_responses) -> TeamRunner:
    coder = agent(settings, *coder_responses)
    pm = PMAgent(settings, model=ScriptedChatModel(responses=list(pm_responses)))
    return TeamRunner(settings, pm=pm, coder=coder)


def happy_runner(settings) -> TeamRunner:
    return make_runner(settings, [spec_response(make_spec())], *happy(), *happy())


def bad_spec_responses() -> list:
    bad = make_spec().model_dump()
    for task in bad["tasks"]:
        task["covers"] = ["AC1"]
    return [structured("Spec", bad), structured("Spec", bad)]


def finish_only() -> list:
    return [ai("done", tool_call("finish", explanation="nothing", plan=["finish"], entry_point="x.py"))]


def load(runner: TeamRunner, name: str) -> dict:
    return json.loads((runner.run_dir / name).read_text())


def test_evidence_files_written(settings):
    runner = happy_runner(settings)
    runner.run(REQ)
    assert runner.run_dir.parent == settings.state_dir / "runs"
    assert sorted(p.name for p in runner.run_dir.iterdir()) == sorted([
        "requirement.txt", "state-after-pm.json", "handoff-T1.json", "state-after-T1.json",
        "handoff-T2.json", "state-after-T2.json", "final-state.json",
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
    assert load(runner, "state-after-T1.json")["task_status"]["T1"]["status"] == "done"


def test_before_after_handoff(settings):
    runner = happy_runner(settings)
    runner.run(REQ)
    before = load(runner, "state-after-pm.json")
    after = load(runner, "state-after-T1.json")
    assert before["task_status"]["T1"]["status"] == "pending"
    assert after["task_status"]["T1"]["status"] == "done"
    assert after["task_status"]["T2"]["status"] == "pending"
    for path in after["task_status"]["T1"]["files"]:
        assert path in after["code_output"]
    assert {"palindrome.py", "test_palindrome.py"} <= set(after["code_output"])


def test_consolidation_file_when_over_8(settings):
    s = settings.model_copy(update={"max_iterations": 2})
    scripts = [r for _ in range(8) for r in finish_only()]
    runner = make_runner(s, [spec_response(make_spec(11)), RuntimeError("no merge")], *scripts)
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
        d = next(i for i, e in enumerate(events) if e.kind == "task_done" and e.task_id == task_id)
        between = events[h + 1:d]
        assert between and all(e.kind == "coder" and e.task_id == task_id for e in between)
        assert events[d].data["status"] == "done"
    assert events[-1].data["state"] == runner.final_state
    assert events[-1].data["api_check"] is None


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
            list(TeamRunner(settings, pm=runner.pm, coder=runner.coder).stream("r"))
        return False, "FAIL echo"

    events = list(runner.stream(REQ, api_check=api_check))
    assert calls == [(settings.workspace_dir, ["palindrome.py", "test_palindrome.py"])]
    assert "FAIL echo" in (runner.run_dir / "api-check.txt").read_text()
    assert events[-1].data["api_check"] == {"passed": False, "report": "FAIL echo"}
    assert [p.name for p in (settings.state_dir / "runs").iterdir()] == [runner.run_dir.name]


def test_api_check_skipped_when_pm_failed(settings):
    runner = make_runner(settings, bad_spec_responses())
    calls = []

    def api_check(workspace, written):
        calls.append(workspace)
        return True, "PASS"

    events = list(runner.stream(REQ, api_check=api_check))
    assert calls == []
    assert not (runner.run_dir / "api-check.txt").exists()
    assert "error" in next(e for e in events if e.kind == "spec").data
    assert events[-1].data["state"]["status"] == "failed"
    assert events[-1].data["api_check"] is None


def test_pm_setup_failure_writes_failed_state(settings):
    class NoStructured(ScriptedChatModel):
        def with_structured_output(self, *a, **kw):
            raise NotImplementedError("structured output not supported")

    runner = TeamRunner(settings, pm=PMAgent(settings, model=NoStructured(responses=[])), coder=agent(settings))
    state = runner.run("Build a REST API")
    assert state["status"] == "failed"
    assert "not supported" in state["error"]
    assert load(runner, "final-state.json")["status"] == "failed"

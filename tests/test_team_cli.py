import json

from app import team_cli
from app.api_check import CheckResult
from app.team_cli import DEMO_API, DEMO_LARGE, main
from agents.pm_agent import PMAgent
from orchestration.project_state import Spec
from orchestration.team_runner import TeamRunner
from tests.fakes import ScriptedChatModel, agent, ai, happy, make_spec, spec_response, tool_call
from tools.workspace import RunLock


def make_runner(settings, *coder_responses) -> TeamRunner:
    pm = PMAgent(settings, model=ScriptedChatModel(responses=[spec_response(make_spec())]))
    return TeamRunner(settings, pm=pm, coder=agent(settings, *coder_responses))


def test_json_output_is_final_state(settings, capsys):
    r = make_runner(settings, *happy(), *happy())
    assert main(["--json", "Build X"], runner=r) == 0
    state = json.loads(capsys.readouterr().out)
    assert state["status"] == "completed"
    Spec.model_validate(state["spec"])


def test_usage_errors(settings, capsys):
    assert main([]) == 2
    assert main(["x", "--demo", "api"]) == 2


def test_partial_exits_1(settings):
    r = make_runner(settings, RuntimeError("boom"), *happy())
    assert main(["Build X"], runner=r) == 1
    assert r.final_state["status"] == "partial"


def test_busy_exits_2(settings, capsys):
    r = make_runner(settings)
    with RunLock(settings.state_dir):
        assert main(["Build X"], runner=r) == 2
    assert "workspace busy" in capsys.readouterr().err


def test_prints_run_dir_and_task_headers(settings, capsys):
    r = make_runner(settings, *happy(), *happy())
    assert main(["Build X"], runner=r) == 0
    out = capsys.readouterr().out
    assert str(r.run_dir) in out.replace("\n", "")
    assert "T1/2" in out and "T2/2" in out


def test_reset_memory_only(settings):
    (settings.workspace_dir).mkdir(parents=True, exist_ok=True)
    (settings.workspace_dir / "old.py").write_text("x = 1")
    r = make_runner(settings)
    assert main(["--reset-memory"], runner=r) == 0
    assert not any(settings.workspace_dir.iterdir())


def app_scripts(path: str) -> list:
    """A one-task script that writes a FastAPI app at `path`, then finishes."""
    return [
        ai("1. write app\n2. write test\n3. run",
           tool_call("write_file", path=path, content="from fastapi import FastAPI\napp = FastAPI()\n"),
           tool_call("write_file", path="test_ok.py", content="from pkg.main import app\n\ndef test_ok():\n    assert app is not None\n"),
           tool_call("exec_python", path="test_ok.py")),
        ai("done", tool_call("finish", explanation="app", plan=["write", "test"], entry_point="test_ok.py")),
    ]


def test_demo_api_runs_fixture(settings, monkeypatch):
    seen = {}

    def fake_check(spec, app_dir, *a, **kw):
        seen["locked"] = RunLock.is_running(settings.state_dir)
        seen["spec"] = spec
        return [CheckResult(name="GET /health", passed=False, detail="nope")]

    monkeypatch.setattr(team_cli, "check_api", fake_check)
    r = make_runner(settings, *app_scripts("pkg/main.py"), *happy())
    assert main(["--demo", "api"], runner=r) == 1
    assert r.final_state["status"] == "completed"
    assert "FAIL" in (r.run_dir / "api-check.txt").read_text()
    assert seen["locked"] is True
    assert seen["spec"] == "pkg.main:app"


def test_demo_api_without_app_fails_without_guessing(settings, monkeypatch):
    calls = []
    monkeypatch.setattr(team_cli, "check_api", lambda *a, **kw: calls.append(a) or [])
    r = make_runner(settings, *happy(), *happy())
    assert main(["--demo", "api"], runner=r) == 1
    assert calls == []
    report = (r.run_dir / "api-check.txt").read_text()
    assert report == f"FAIL find app: no file this run delivered assigns app = FastAPI(...) in {settings.workspace_dir}"


def test_demo_api_ignores_leftover_app(settings, monkeypatch):
    # An app from an earlier run is in the workspace, but this run delivered none: the check fails
    calls = []
    monkeypatch.setattr(team_cli, "check_api", lambda *a, **kw: calls.append(a) or [])
    settings.workspace_dir.mkdir(parents=True, exist_ok=True)
    (settings.workspace_dir / "main.py").write_text("from fastapi import FastAPI\napp = FastAPI()\n")
    r = make_runner(settings, *happy(), *happy())
    assert main(["--demo", "api"], runner=r) == 1
    assert calls == []
    assert (r.run_dir / "api-check.txt").read_text().startswith("FAIL find app:")

def test_demo_api_text():
    assert DEMO_API.startswith("Build a REST API with two endpoints: GET /health (returns {status: ok})")
    assert "422" in DEMO_API


def test_demo_large_lists_many_features():
    assert len(DEMO_LARGE) > 200


def test_summary_lists_only_executed_tasks():
    # The summary lists the tasks in data["tasks"], as finalize judges them; records for ids
    # outside that list are ignored
    from io import StringIO

    from rich.console import Console

    from app.render import render_team_summary

    out = StringIO()
    data = {
        "tasks": [{"task_id": "T1"}],
        "task_status": {
            "T1": {"task_id": "T1", "status": "done", "iterations": 2, "files": ["a.py"]},
            "T9": {"task_id": "T9", "status": "pending"},
        },
        "status": "completed",
    }
    render_team_summary(Console(file=out, width=100), data)
    text = out.getvalue()
    assert "T1" in text and "done" in text
    assert "T9" not in text and "pending" not in text

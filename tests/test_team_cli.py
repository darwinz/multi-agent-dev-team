import json

from app import team_cli
from app.api_check import CheckResult
from app.team_cli import DEMO_API, DEMO_LARGE, main
from agents.pm_agent import PMAgent
from orchestration.project_state import Spec
from orchestration.team_runner import TeamRunner
from tests.fakes import ScriptedChatModel, agent, happy, make_spec, spec_response
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


def test_demo_api_runs_fixture(settings, monkeypatch):
    seen = {}

    def fake_check(spec, app_dir, *a, **kw):
        seen["locked"] = RunLock.is_running(settings.state_dir)
        return [CheckResult(name="GET /health", passed=False, detail="nope")]

    monkeypatch.setattr(team_cli, "check_api", fake_check)
    r = make_runner(settings, *happy(), *happy())
    assert main(["--demo", "api"], runner=r) == 1
    assert r.final_state["status"] == "completed"
    assert "FAIL" in (r.run_dir / "api-check.txt").read_text()
    assert seen["locked"] is True


def test_demo_api_text():
    assert DEMO_API.startswith("Build a REST API with two endpoints: GET /health (returns {status: ok})")
    assert "422" in DEMO_API


def test_demo_large_lists_many_features():
    assert len(DEMO_LARGE) > 200

import json

import pytest

from agents.config import Settings
from app import team_cli
from app.api_check import CheckResult
from app.team_cli import DEMO_API, DEMO_BST, DEMO_LARGE, DEMO_LINKED_LIST, main
from orchestration.project_state import Spec
from orchestration.team_runner import TeamRunner
from tests.fakes import (
    PAL,
    TEST,
    ai,
    critique_response,
    make_spec,
    qa_suite_response,
    revise_happy,
    spec_response,
    task_scripts,
    tool_call,
)
from tests.scenarios import (
    QA_BST_MANIFEST,
    bst_scripts,
    coder_run,
    feedback_scripts,
    nonconverging_scripts,
    passing_scripts,
    scripted_runner,
)
from tools.workspace import RunLock


@pytest.fixture
def settings(tmp_path) -> Settings:
    # These tests run real QA pytest through the team graph; 15 s keeps them stable under load
    return Settings.from_env(project_dir=tmp_path, exec_timeout_s=15)


def make_runner(settings, coder: list, qa: list) -> TeamRunner:
    return scripted_runner(settings, ([spec_response(make_spec())], coder, qa))


def two_tasks(settings) -> TeamRunner:
    (c1, q1), (c2, q2) = task_scripts(), task_scripts()
    return make_runner(settings, c1 + c2, q1 + q2)


def test_json_output_is_final_state(settings, capsys):
    r = two_tasks(settings)
    assert main(["--json", "Build X"], runner=r) == 0
    state = json.loads(capsys.readouterr().out)
    assert state["status"] == "completed"
    Spec.model_validate(state["spec"])


def test_usage_errors(settings, capsys):
    assert main([]) == 2
    assert main(["x", "--demo", "api"]) == 2


def test_partial_exits_1(settings):
    # T1: the Coder errors and the critique fails twice (no revision); QA can't write a suite. T2 passes.
    c2, q2 = task_scripts()
    coder = [RuntimeError("boom"), RuntimeError("no critique"), RuntimeError("again"), *c2]
    r = make_runner(settings, coder, [RuntimeError("no suite"), *q2])
    assert main(["Build X"], runner=r) == 1
    assert r.final_state["status"] == "partial"
    assert r.final_state["task_status"]["T1"]["status"] == "failed"
    assert r.final_state["task_status"]["T2"]["status"] == "done"


def test_busy_exits_2(settings, capsys):
    r = make_runner(settings, [], [])
    with RunLock(settings.state_dir):
        assert main(["Build X"], runner=r) == 2
    assert "workspace busy" in capsys.readouterr().err


def test_prints_run_dir_and_task_headers(settings, capsys):
    r = two_tasks(settings)
    assert main(["Build X"], runner=r) == 0
    out = capsys.readouterr().out
    assert str(r.run_dir) in out.replace("\n", "")
    assert "T1/2" in out and "T2/2" in out


def test_transcript_shows_a2a_and_qa(settings, capsys):
    r = two_tasks(settings)
    assert main(["Build X"], runner=r) == 0
    out = capsys.readouterr().out
    assert "A2A coder → qa review_request" in out
    assert "A2A qa → coder approved" in out
    assert "self-review" in out
    assert "QA round 1" in out
    for column in ("qa rounds", "approved"):
        assert column in out


def test_reset_memory_only(settings):
    (settings.workspace_dir).mkdir(parents=True, exist_ok=True)
    (settings.workspace_dir / "old.py").write_text("x = 1")
    r = make_runner(settings, [], [])
    assert main(["--reset-memory"], runner=r) == 0
    assert not any(settings.workspace_dir.iterdir())


@pytest.mark.parametrize(("argv", "env"), [
    (["--qa-max-iterations", "0"], None),
    (["--qa-max-iterations", "6"], None),
    (["--qa-max-iterations", "abc"], None),
    ([], "0"),
    ([], "6"),
    ([], "abc"),
], ids=["flag-0", "flag-6", "flag-abc", "env-0", "env-6", "env-abc"])
def test_bad_qa_max_iterations_exits_2(tmp_path, monkeypatch, capsys, argv, env):
    monkeypatch.setenv("CODER_PROJECT_DIR", str(tmp_path))
    if env is None:
        monkeypatch.delenv("QA_MAX_ITERATIONS", raising=False)
    else:
        monkeypatch.setenv("QA_MAX_ITERATIONS", env)
    built = []
    monkeypatch.setattr(team_cli, "TeamRunner", lambda settings: built.append(settings))
    assert main([*argv, "Build X"]) == 2
    assert "team: invalid setting qa_max_iterations:" in capsys.readouterr().err
    assert built == []


def test_qa_max_iterations_flag_reaches_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("CODER_PROJECT_DIR", str(tmp_path))
    monkeypatch.setenv("QA_MAX_ITERATIONS", "4")
    built = []

    class FakeRunner:
        def __init__(self, settings):
            built.append(settings)

        def reset_memory(self):
            pass

    monkeypatch.setattr(team_cli, "TeamRunner", FakeRunner)
    assert main(["--reset-memory", "--qa-max-iterations", "2"]) == 0
    assert main(["--reset-memory"]) == 0
    assert [s.qa_max_iterations for s in built] == [2, 4]


def test_logging_configured_before_runner_is_built(tmp_path, monkeypatch):
    """TeamRunner's startup (Chroma connection retries) already logs: those lines must be JSON too."""
    monkeypatch.setenv("CODER_PROJECT_DIR", str(tmp_path))
    order = []

    class FakeRunner:
        def __init__(self, settings):
            order.append("runner")

        def reset_memory(self):
            pass

    monkeypatch.setattr(team_cli, "TeamRunner", FakeRunner)
    monkeypatch.setattr(team_cli, "configure_logging", lambda level: order.append("logging"))
    assert main(["--reset-memory"]) == 0
    assert order[:2] == ["logging", "runner"]


def app_scripts(path: str) -> list:
    """A Coder delivery that writes a FastAPI app at `path` plus the palindrome module, then finishes."""
    return [
        ai("1. write app and module\n2. write test\n3. run",
           tool_call("write_file", path=path, content="from fastapi import FastAPI\napp = FastAPI()\n"),
           tool_call("write_file", path="palindrome.py", content=PAL),
           tool_call("write_file", path="test_palindrome.py", content=TEST),
           tool_call("exec_python", path="test_palindrome.py")),
        ai("done", tool_call("finish", explanation="app", plan=["write", "test"], entry_point="test_palindrome.py")),
    ]


def test_demo_api_runs_fixture(settings, monkeypatch):
    seen = {}

    def fake_check(spec, app_dir, *a, **kw):
        seen["locked"] = RunLock.is_running(settings.state_dir)
        seen["spec"] = spec
        return [CheckResult(name="GET /health", passed=False, detail="nope")]

    monkeypatch.setattr(team_cli, "check_api", fake_check)
    # T1 delivers the app (and the palindrome module QA tests); T2 is a plain palindrome task
    c2, q2 = task_scripts()
    c1 = [*app_scripts("pkg/main.py"), critique_response(), *revise_happy()]
    r = make_runner(settings, c1 + c2, [qa_suite_response(), *q2])
    assert main(["--demo", "api"], runner=r) == 1
    assert r.final_state["status"] == "completed"
    assert "FAIL" in (r.run_dir / "api-check.txt").read_text()
    assert seen["locked"] is True
    assert seen["spec"] == "pkg.main:app"


def test_demo_api_without_app_fails_without_guessing(settings, monkeypatch):
    calls = []
    monkeypatch.setattr(team_cli, "check_api", lambda *a, **kw: calls.append(a) or [])
    r = two_tasks(settings)
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
    r = two_tasks(settings)
    assert main(["--demo", "api"], runner=r) == 1
    assert calls == []
    assert (r.run_dir / "api-check.txt").read_text().startswith("FAIL find app:")


def test_demo_linked_list_feedback_run_passes(settings, capsys):
    r = scripted_runner(settings, feedback_scripts())
    assert main(["--demo", "linked-list"], runner=r) == 0
    assert (r.run_dir / "requirement.txt").read_text() == DEMO_LINKED_LIST
    report = (r.run_dir / "demo-check.txt").read_text()
    assert len(report.splitlines()) == 3 and all(line.startswith("PASS") for line in report.splitlines())
    out = capsys.readouterr().out
    assert "Linked-list check: PASSED" in out
    assert "A2A qa → coder fix_instruction" in out and "QA round 2" in out


def test_demo_linked_list_nonconverging_exits_1(settings, monkeypatch):
    monkeypatch.setenv("QA_MAX_ITERATIONS", "2")
    s = Settings.from_env(project_dir=settings.project_dir, exec_timeout_s=15)
    assert s.qa_max_iterations == 2
    r = scripted_runner(s, nonconverging_scripts(2))
    assert main(["--demo", "linked-list"], runner=r) == 1
    assert r.final_state["status"] == "partial"
    assert (r.run_dir / "demo-check.txt").read_text().startswith("FAIL")


def test_demo_bst_run_passes_supplied_fixture(settings, capsys):
    r = scripted_runner(settings, bst_scripts())
    assert main(["--demo", "bst"], runner=r) == 0
    assert (r.run_dir / "requirement.txt").read_text() == DEMO_BST
    report = (r.run_dir / "demo-check.txt").read_text()
    assert "PASS supplied BST fixture" in report
    assert "BST fixture check: PASSED" in capsys.readouterr().out


def test_demo_bst_exits_1_when_fixture_fails(settings):
    # inorder returns insertion order; QA's weak suite only inserts ascending values, so QA approves
    # and the run completes, but the supplied fixture's random-order case fails the demo
    unsorted = ("class BinarySearchTree:\n    def __init__(self):\n        self.items = []\n\n"
                "    def insert(self, value):\n        if value not in self.items:\n            self.items.append(value)\n\n"
                "    def search(self, value):\n        return value in self.items\n\n"
                "    def inorder(self):\n        return list(self.items)\n")
    weak = ("from bst import BinarySearchTree\n\n\ndef make():\n    t = BinarySearchTree()\n"
            "    for v in (1, 2, 2, 3):\n        t.insert(v)\n    return t\n\n\n"
            "def test_insert():\n    assert BinarySearchTree().insert(1) is None\n\n\n"
            "def test_search():\n    assert make().search(2) and not make().search(9)\n\n\n"
            "def test_inorder():\n    assert make().inorder() == [1, 2, 3]\n\n\n"
            "def test_duplicates():\n    assert len(make().inorder()) == 3\n")
    pm, _, _ = bst_scripts()
    coder = [*coder_run({"bst.py": unsorted}, "bst.py", "BST"),
             critique_response(("edge_case", "bst.py", "no docs", "add a docstring")),
             *coder_run({"bst.py": '"""BST."""\n' + unsorted}, "bst.py", "Documented")]
    r = scripted_runner(settings, (pm, coder, [qa_suite_response(weak, QA_BST_MANIFEST)]))
    assert main(["--demo", "bst"], runner=r) == 1
    assert r.final_state["status"] == "completed"
    assert (r.run_dir / "demo-check.txt").read_text().splitlines()[1].startswith("FAIL supplied BST fixture")


def test_demo_bst_text():
    assert DEMO_BST.startswith("Build a Python module that implements a binary search tree with insert, search, "
                               "and in-order traversal methods, with full test coverage.\n\n"
                               "Interface (the supplied fixture's contract): ")
    for needle in ("bst.py", "BinarySearchTree", "insert(value) -> None", "search(value) -> bool",
                   "inorder() -> list", "ascending", "duplicate", "mutually comparable"):
        assert needle in DEMO_BST


def test_demo_api_text():
    assert DEMO_API.startswith("Build a REST API with two endpoints: GET /health (returns {status: ok})")
    assert "422" in DEMO_API


def test_demo_large_lists_many_features():
    assert len(DEMO_LARGE) > 200


def test_demo_linked_list_text():
    assert DEMO_LINKED_LIST == ("Implement a Python module that contains a linked list class with insert, delete, "
                                "search, and reverse methods. All methods must be covered by tests.")


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
            "T1": {"task_id": "T1", "status": "done", "iterations": 2, "files": ["a.py"], "qa_rounds": 2,
                   "qa_outcome": "approved"},
            "T9": {"task_id": "T9", "status": "pending"},
        },
        "status": "completed",
    }
    render_team_summary(Console(file=out, width=120), data)
    text = out.getvalue()
    assert "T1" in text and "done" in text and "approved" in text
    assert "qa rounds" in text and "iterations" in text
    assert "T9" not in text and "pending" not in text


def test_render_a2a_lines():
    from io import StringIO

    from rich.console import Console

    from app.render import render_a2a

    msg = {"sender": "coder", "receiver": "qa", "intent": "review_request", "correlation_id": "1a2b3c4d" + "0" * 24,
           "payload": {"task_id": "T1", "round": 1, "files": {"a.py": "x", "b.py": "y"}}}
    report = {"sender": "qa", "receiver": "coder", "intent": "final_report", "correlation_id": "f" * 32,
              "payload": {"task_id": "T1", "round": 2, "reason": "max_iterations", "unresolved": [{}], "summary": ""}}
    out = StringIO()
    console = Console(file=out, width=200)
    render_a2a(console, {"event": "sent", "message": msg, "reason": None, "at": "now"})
    render_a2a(console, {"event": "received", "message": msg, "reason": None, "at": "now"})
    render_a2a(console, {"event": "rejected", "message": msg, "reason": "stale", "at": "now"})
    render_a2a(console, {"event": "sent", "message": report, "reason": None, "at": "now"})
    assert out.getvalue().splitlines() == [
        "A2A coder → qa review_request #1a2b3c4d round 1 · 2 files",
        "A2A qa received review_request #1a2b3c4d",
        "A2A qa rejected review_request #1a2b3c4d from coder: stale",
        "A2A qa → coder final_report #ffffffff round 2 · max_iterations · 1 unresolved",
    ]


def test_team_exits_2_when_chroma_unreachable(tmp_path, monkeypatch, capsys):  # Review Focus 3
    monkeypatch.setenv("CHROMA_HOST", "127.0.0.1")
    monkeypatch.setenv("CHROMA_PORT", "9")  # discard port: nothing listens, so the connect is refused at once
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("CODER_PROJECT_DIR", str(tmp_path))
    assert team_cli.main(["do x"]) == 2
    err = capsys.readouterr().err
    assert "cannot run" in err and "Chroma" in err and "127.0.0.1:9" in err and "Traceback" not in err


def test_prints_cost_report_path(settings, capsys):
    r = two_tasks(settings)
    assert main(["Build X"], runner=r) == 0
    out = capsys.readouterr().out.replace("\n", "")
    assert f"cost report: {r.cost_report_path}" in out
    assert json.loads(r.cost_report_path.read_text())["outcome"] == "success"


def test_team_ctrl_c_exits_130_with_report(tmp_path, monkeypatch, capsys):  # Review Focus 1
    monkeypatch.setenv("CODER_PROJECT_DIR", str(tmp_path))
    monkeypatch.delenv("CHROMA_HOST", raising=False)
    monkeypatch.setattr(team_cli, "provider_prerequisite", lambda model: None)  # scripted models: no key needed
    built = []

    def offline_runner(settings):  # the real Settings path, but scripted models: no network call is possible
        runner = scripted_runner(settings, passing_scripts())
        runner.qa.review = lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt())
        built.append(runner)
        return runner

    monkeypatch.setattr(team_cli, "TeamRunner", offline_runner)
    assert main(["Build X"]) == 130
    err = capsys.readouterr().err
    assert "interrupted" in err and "Traceback" not in err
    (runner,) = built
    report = json.loads(runner.cost_report_path.read_text())
    assert runner.cost_report_path.parent == tmp_path / "docs" / "cost_reports"
    assert report["outcome"] == "aborted" and report["error"]["type"] == "KeyboardInterrupt"
    assert not RunLock.is_running(runner.settings.state_dir)


def test_prints_cost_report_unavailable(settings, capsys):
    r = two_tasks(settings)
    original = TeamRunner.stream

    def stream(self, *a, **k):  # as if neither copy of the report could be written
        yield from original(self, *a, **k)
        self.cost_report_path = None

    r.stream = stream.__get__(r)
    assert main(["Build X"], runner=r) == 0
    assert "cost report: unavailable (see events.jsonl)" in capsys.readouterr().out.replace("\n", "")

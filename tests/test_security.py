import json
import logging
from types import SimpleNamespace

import pytest

from orchestration.review import Issue
from tests.fakes import ScriptedChatModel, agent, ai, make_spec, qa_agent, qa_suite_response, tool_call
from tests.scenarios import passing_scripts, scripted_runner
from tests.test_qa_agent import STACK_TESTS, handoff, stack_suite
from tools.exec_python import child_env, exec_python
from tools.workspace import Workspace


def test_generated_code_cannot_see_api_keys(settings, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-leak")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2-should-not-leak")
    ws = Workspace(settings.workspace_dir)
    out = exec_python(ws, record_dir=settings.state_dir / "exec", mode="script",
                      code="import os, json\nprint(json.dumps(sorted(os.environ)))")
    keys = json.loads(out.stdout)
    assert "ANTHROPIC_API_KEY" not in keys and "LANGSMITH_API_KEY" not in keys and "PATH" in keys


def test_child_env_is_allowlist_only(tmp_path, monkeypatch):
    monkeypatch.setenv("SOME_TOKEN", "x" * 20)
    names = set(child_env(tmp_path))
    assert "SOME_TOKEN" not in names


def test_child_env_keeps_locale_and_pycache_prefix(tmp_path, monkeypatch):
    monkeypatch.setenv("LC_CTYPE", "C.UTF-8")
    env = child_env(tmp_path)
    assert "LC_CTYPE" in set(env) and "PYTHONPYCACHEPREFIX" in set(env)
    assert env["LC_CTYPE"] == "C.UTF-8"
    assert env["PYTHONPYCACHEPREFIX"] == str(tmp_path / "pycache")


# --- invalid-output probe (spec 5.7): bad model output never reaches a side-effecting tool -------------

REQ = "Build a linked list"


@pytest.fixture
def spies(monkeypatch):
    """Recorders on every write_file / exec_python the agents look up at call time; they call through."""
    import agents.qa_agent as qa_mod
    import tools.dispatch as dispatch_mod

    calls: list[tuple[str, tuple]] = []

    def spy(name, real):
        def recorder(*args, **kwargs):
            calls.append((name, args))
            return real(*args, **kwargs)
        return recorder

    for mod in (dispatch_mod, qa_mod):
        for name in ("write_file", "exec_python"):
            monkeypatch.setattr(mod, name, spy(f"{mod.__name__}.{name}", getattr(mod, name)))
    return SimpleNamespace(calls=calls)


def test_invalid_pm_spec_never_reaches_tools(settings, spies):
    runner = scripted_runner(settings, ([ai("not a spec"), ai("still not")], [], []))
    state = runner.run(REQ)
    assert state["status"] == "failed" and "invalid spec twice" in state["error"] and spies.calls == []


def test_invalid_qa_suite_never_executes(settings, spies):
    # Coder delivers + reflects; QA returns two invalid suites (a manifest name that is not in the code)
    pm, coder, _ = passing_scripts()
    bad = qa_suite_response(STACK_TESTS, (("test_nope", 1),))
    state = scripted_runner(settings, (pm, coder, [bad, bad])).run(REQ)
    assert not [c for c in spies.calls if c[0].startswith("agents.qa_agent.exec_python")]
    assert [t["qa_outcome"] for t in state["task_status"].values()] == ["qa_error"]


def test_invalid_tool_call_rejected_before_write(settings, spies):
    a = agent(settings,
              ai("escape", tool_call("write_file", path="../escape.py", content="x")),
              ai("no content", tool_call("write_file", path="ok.py")),
              ai("done", tool_call("finish", explanation="e", plan=["p"])))
    a.run("write something")
    assert not (settings.workspace_dir.parent / "escape.py").exists()
    # the missing-content call dies in ARG_MODELS; the escaping path reaches write_file only to be refused by Workspace
    assert [c[1][1:] for c in spies.calls if c[0].endswith("dispatch.write_file")] == [("../escape.py", "x")]
    observations = [str(m.content) for m in a.model.inner.calls[-1] if getattr(m, "type", "") == "tool"]
    assert len(observations) == 2 and all(o.startswith("error:") for o in observations)
    assert not (settings.workspace_dir / "ok.py").exists()


def _warnings(caplog, event):
    return [r for r in caplog.records if r.getMessage() == event and r.levelno == logging.WARNING]


def test_consolidation_fallback_logs_warning(settings, caplog):
    from agents.pm_agent import PMAgent

    model = ScriptedChatModel(responses=[RuntimeError("no merge")])
    with caplog.at_level(logging.WARNING, logger="mad"):
        _, cons = PMAgent(settings, model=model).consolidate(make_spec(11, 4))
    assert cons.method == "deterministic"
    (rec,) = _warnings(caplog, "pm_consolidation_fallback")
    assert rec.fields["error_class"] == "RuntimeError" and "no merge" in rec.fields["message"]


def test_fix_plan_and_audit_fallbacks_log_warning(settings, caplog):
    qa = qa_agent(settings, RuntimeError("no plan"), RuntimeError("no audit"))
    issue = Issue(severity="High", test="test_push_then_pop", criterion=None, location="stack.py:1",
                  assertion="boom", fix="orig")
    with caplog.at_level(logging.WARNING, logger="mad"):
        assert qa._fixes([issue], {}) == [issue]
        _suite, corrections = qa._check_tests(handoff(), stack_suite(), [issue], {})
    assert corrections == []
    (fix_rec,) = _warnings(caplog, "qa_fix_plan_fallback")
    (audit_rec,) = _warnings(caplog, "qa_audit_fallback")
    assert fix_rec.fields["error_class"] == "RuntimeError" and "no plan" in fix_rec.fields["message"]
    assert audit_rec.fields["error_class"] == "RuntimeError" and "no audit" in audit_rec.fields["message"]

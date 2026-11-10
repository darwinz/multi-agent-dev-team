import errno

import pytest

from observability.context import RunContext, run_scope
from tools.dispatch import ToolContext, run_action
from tools.exec_python import ExecOutcome
from tools.specs import TOOL_SPECS
from tools.workspace import ToolError, Workspace

FAKE_OUTCOME = ExecOutcome("s.py", "script", 0, "", "", False, 0.1)


def ctx(tmp_path):
    return ToolContext(Workspace(tmp_path / "ws"), 5.0, tmp_path / "rec")


def test_specs_cover_all_tools():
    assert [s["name"] for s in TOOL_SPECS] == ["read_file", "write_file", "exec_python", "finish"]
    assert all(s["parameters"]["type"] == "object" for s in TOOL_SPECS)


def test_write_then_exec(tmp_path):
    c = ctx(tmp_path)
    w = run_action(c, "write_file", {"path": "a.py", "content": "print(1)"})
    assert w.write.path == "a.py" and w.observation.startswith("wrote a.py")
    e = run_action(c, "exec_python", {"path": "a.py"})
    assert e.execution.ok and "exit_code=0" in e.observation


def test_default_timeout_comes_from_context(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr("tools.dispatch.exec_python", lambda ws, **kw: seen.update(kw) or FAKE_OUTCOME)
    run_action(ctx(tmp_path), "exec_python", {"code": "print(1)"})
    assert seen["timeout"] == 5.0


@pytest.mark.parametrize("name,args,msg", [
    ("write_file", {"path": "a.py"}, "invalid arguments for write_file"),
    ("rm_rf", {}, "unknown tool: rm_rf"),
])
def test_bad_calls_are_tool_errors(tmp_path, name, args, msg):
    with pytest.raises(ToolError, match=msg):
        run_action(ctx(tmp_path), name, args)


def test_per_call_timeout_cannot_exceed_configured(tmp_path, monkeypatch):
    # the operator's --timeout is a ceiling: the model passed timeout=5 against --timeout 2 in a live probe
    seen = {}
    monkeypatch.setattr("tools.dispatch.exec_python", lambda ws, **kw: seen.update(kw) or FAKE_OUTCOME)
    c = ToolContext(Workspace(tmp_path / "ws"), 2.0, tmp_path / "rec")
    run_action(c, "exec_python", {"code": "print(1)", "timeout": 5})
    assert seen["timeout"] == 2.0
    run_action(c, "exec_python", {"code": "print(1)", "timeout": 1})
    assert seen["timeout"] == 1.0


@pytest.mark.parametrize("args", [
    {"path": "workspace", "content": "x"},  # resolves to the workspace dir itself
    {"path": "s.py", "content": "\ud800"},  # lone surrogate: not encodable
])
def test_os_and_encoding_errors_become_tool_errors(tmp_path, args):  # review I4
    with pytest.raises(ToolError, match="write_file failed"):
        run_action(ctx(tmp_path), "write_file", args)


def test_large_timeout_request_is_clamped_not_rejected(tmp_path, monkeypatch):  # review M5
    seen = {}
    monkeypatch.setattr("tools.dispatch.exec_python", lambda ws, **kw: seen.update(kw) or FAKE_OUTCOME)
    run_action(ctx(tmp_path), "exec_python", {"code": "print(1)", "timeout": 90})
    assert seen["timeout"] == 5.0


def test_files_written_by_executed_code_are_reported(tmp_path):  # review M7
    out = run_action(ctx(tmp_path), "exec_python", {"code": "open('gen.py', 'w').write('X = 1\\n')"})
    assert [w.path for w in out.side_writes] == ["gen.py"]
    assert "files changed by this run: gen.py" in out.observation


def flaky_write(fails: int):
    """A write_file stand-in that raises EBUSY (a transient OS error) `fails` times, then writes."""
    from tools.file_tools import write_file
    calls = {"n": 0}

    def fn(ws, path, content):
        calls["n"] += 1
        if calls["n"] <= fails:
            raise OSError(errno.EBUSY, "Device or resource busy")
        return write_file(ws, path, content)
    return fn, calls


def test_transient_os_error_retried_then_succeeds(tmp_path, monkeypatch):
    fn, calls = flaky_write(1)
    monkeypatch.setattr("tools.dispatch.write_file", fn)
    run = RunContext(run_id="r")
    with run_scope(run):
        out = run_action(ctx(tmp_path), "write_file", {"path": "a.py", "content": "x = 1\n"})
    assert out.write.path == "a.py" and calls["n"] == 2
    assert [(t.tool, t.ok, t.attempts) for t in run.tools] == [("write_file", True, 2)]


def test_persistent_tool_failure_becomes_tool_error_after_bound(tmp_path, monkeypatch):
    fn, calls = flaky_write(4)
    monkeypatch.setattr("tools.dispatch.write_file", fn)
    run = RunContext(run_id="r")
    with run_scope(run), pytest.raises(ToolError, match="write_file unavailable: OSError"):
        run_action(ctx(tmp_path), "write_file", {"path": "a.py", "content": "x = 1\n"})
    assert calls["n"] == 4  # the original attempt plus exactly 3 retries
    assert [(t.tool, t.ok, t.attempts) for t in run.tools] == [("write_file", False, 4)]


def test_open_tools_breaker_is_a_tool_error(tmp_path, monkeypatch):
    fn, calls = flaky_write(100)
    monkeypatch.setattr("tools.dispatch.write_file", fn)
    for _ in range(3):  # BREAKER_FAILURE_THRESHOLD exhausted calls open the tools breaker
        with pytest.raises(ToolError, match="unavailable"):
            run_action(ctx(tmp_path), "write_file", {"path": "a.py", "content": "x"})
    n = calls["n"]
    with pytest.raises(ToolError, match="write_file unavailable"):
        run_action(ctx(tmp_path), "write_file", {"path": "a.py", "content": "x"})
    assert calls["n"] == n  # rejected without an attempt


def test_permanent_tool_error_not_retried_and_type_kept(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fn(ws, path, content):
        calls["n"] += 1
        raise IsADirectoryError(errno.EISDIR, "Is a directory")
    monkeypatch.setattr("tools.dispatch.write_file", fn)
    run = RunContext(run_id="r")
    with run_scope(run), pytest.raises(ToolError, match="write_file failed: IsADirectoryError"):
        run_action(ctx(tmp_path), "write_file", {"path": "a.py", "content": "x"})
    assert calls["n"] == 1
    assert [(t.ok, t.attempts) for t in run.tools] == [(False, 1)]


def test_program_failure_is_a_result_not_retried(tmp_path, monkeypatch):
    from tools.exec_python import exec_python
    calls = {"n": 0}

    def counted(*a, **kw):
        calls["n"] += 1
        return exec_python(*a, **kw)
    monkeypatch.setattr("tools.dispatch.exec_python", counted)
    run = RunContext(run_id="r")
    with run_scope(run):
        out = run_action(ctx(tmp_path), "exec_python", {"code": "raise SystemExit(3)"})
    assert out.execution.exit_code == 3 and "exit_code=3" in out.observation
    assert calls["n"] == 1
    assert [(t.tool, t.ok, t.attempts) for t in run.tools] == [("exec_python", True, 1)]

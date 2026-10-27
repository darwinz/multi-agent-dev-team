import pytest

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

import json

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
    assert "SOME_TOKEN" not in child_env(tmp_path)


def test_child_env_keeps_locale_and_pycache_prefix(tmp_path, monkeypatch):
    monkeypatch.setenv("LC_CTYPE", "C.UTF-8")
    env = child_env(tmp_path)
    assert env["LC_CTYPE"] == "C.UTF-8"
    assert env["PYTHONPYCACHEPREFIX"] == str(tmp_path / "pycache")

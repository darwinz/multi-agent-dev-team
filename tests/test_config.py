import pytest
from pydantic import ValidationError

from agents.config import Settings


def test_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("CODER_MODEL", raising=False)
    s = Settings.from_env(project_dir=tmp_path)
    assert s.model == "anthropic:claude-sonnet-5-5"
    assert s.max_iterations == 10 and s.exec_timeout_s == 10.0 and s.window_turns == 10
    assert s.workspace_dir == tmp_path / "workspace"
    assert s.state_dir == tmp_path / ".coder" and s.chroma_dir == tmp_path / ".chroma"


def test_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("CODER_MAX_ITERATIONS", "3")
    monkeypatch.setenv("EXEC_TIMEOUT_SECONDS", "2.5")
    s = Settings.from_env(project_dir=tmp_path)
    assert (s.max_iterations, s.exec_timeout_s) == (3, 2.5)


@pytest.mark.parametrize("bad", [0, 11, 50])
def test_max_iterations_never_above_cap(tmp_path, bad):
    with pytest.raises(ValidationError, match="between 1 and 10"):
        Settings.from_env(project_dir=tmp_path, max_iterations=bad)


def test_timeout_must_be_positive(tmp_path):
    with pytest.raises(ValidationError):
        Settings.from_env(project_dir=tmp_path, exec_timeout_s=0)


def test_dotenv_read_from_project_dir_only(tmp_path, monkeypatch):
    monkeypatch.delenv("CODER_MODEL", raising=False)
    (tmp_path / ".env").write_text("CODER_MODEL=openai:gpt-4o\n")
    assert Settings.from_env(project_dir=tmp_path).model == "openai:gpt-4o"

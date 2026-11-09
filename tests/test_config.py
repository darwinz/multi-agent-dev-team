import pytest
from pydantic import ValidationError

from agents.config import Settings


def test_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("CODER_MODEL", raising=False)
    s = Settings.from_env(project_dir=tmp_path)
    assert s.model == "anthropic:claude-haiku-5-5"
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


def test_qa_max_iterations_default_and_env(tmp_path, monkeypatch):
    monkeypatch.delenv("QA_MAX_ITERATIONS", raising=False)
    assert Settings.from_env(project_dir=tmp_path).qa_max_iterations == 5
    monkeypatch.setenv("QA_MAX_ITERATIONS", "3")
    assert Settings.from_env(project_dir=tmp_path).qa_max_iterations == 3


def test_qa_max_iterations_bounds(tmp_path, monkeypatch):
    monkeypatch.delenv("QA_MAX_ITERATIONS", raising=False)
    for bad in (0, 6):
        with pytest.raises(ValidationError):
            Settings.from_env(project_dir=tmp_path, qa_max_iterations=bad)
    assert Settings.from_env(project_dir=tmp_path, qa_max_iterations=5).qa_max_iterations == 5


def test_role_model_defaults(tmp_path, monkeypatch):
    for v in ("PM_MODEL", "CODER_MODEL", "QA_MODEL"):
        monkeypatch.delenv(v, raising=False)
    s = Settings.from_env(project_dir=tmp_path)
    assert s.model_for("pm") == "anthropic:claude-sonnet-5-5"
    assert s.model_for("coder") == s.model == "anthropic:claude-haiku-5-5"
    assert s.model_for("qa") == "anthropic:claude-haiku-5-5"


def test_role_model_env_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("PM_MODEL", "anthropic:claude-opus-5-5")
    monkeypatch.setenv("CODER_MODEL", "anthropic:claude-sonnet-5-5")
    monkeypatch.setenv("QA_MODEL", "openai:gpt-4o")
    s = Settings.from_env(project_dir=tmp_path)
    assert (s.model_for("pm"), s.model_for("coder"), s.model_for("qa")) == (
        "anthropic:claude-opus-5-5", "anthropic:claude-sonnet-5-5", "openai:gpt-4o")


def test_retry_max_capped_at_3(tmp_path):
    with pytest.raises(ValidationError):
        Settings.from_env(project_dir=tmp_path, retry_max=4)


def test_resilience_defaults(tmp_path, monkeypatch):
    for v in ("RETRY_BASE_DELAY_S", "RETRY_MAX_DELAY_S", "CACHE_TTL_S"):
        monkeypatch.delenv(v, raising=False)
    s = Settings.from_env(project_dir=tmp_path)
    assert (s.retry_max, s.retry_base_delay_s, s.retry_max_delay_s) == (3, 1.0, 30.0)
    assert (s.breaker_failure_threshold, s.breaker_cooldown_s, s.cache_ttl_s) == (3, 30.0, 3600.0)
    assert s.cache_dir == tmp_path / ".coder" / "cache"
    assert s.cost_reports_dir == tmp_path / "docs" / "cost_reports"

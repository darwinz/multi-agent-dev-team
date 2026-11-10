import pytest

from agents.config import Settings
from resilience import runtime as _resilience_runtime


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings.from_env(project_dir=tmp_path, exec_timeout_s=5)


@pytest.fixture(autouse=True)
def _fast_resilience(monkeypatch, request):
    """Mocked tests never sleep on retry back-off or hit the response cache."""
    _resilience_runtime.reset()
    if not request.node.get_closest_marker("live"):  # live tests keep real back-off and caching
        monkeypatch.setenv("RETRY_BASE_DELAY_S", "0")
        monkeypatch.setenv("RETRY_MAX_DELAY_S", "0")
        monkeypatch.setenv("CACHE_TTL_S", "0")
    yield
    _resilience_runtime.reset()

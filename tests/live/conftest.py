import os

import pytest

from agents.config import Settings
from agents.model import provider_prerequisite


@pytest.fixture
def live_settings(tmp_path) -> Settings:
    settings = Settings.from_env(project_dir=tmp_path)
    if reason := provider_prerequisite(settings.model):
        if os.environ.get("REQUIRE_LIVE") == "1":
            pytest.fail(f"live test required but {reason}")
        pytest.skip(reason)
    return settings


@pytest.fixture(autouse=True)
def _real_resilience(monkeypatch):
    """Live runs use the real retry delays and the default cache TTL."""
    for name in ("RETRY_BASE_DELAY_S", "RETRY_MAX_DELAY_S", "CACHE_TTL_S"):
        monkeypatch.delenv(name, raising=False)

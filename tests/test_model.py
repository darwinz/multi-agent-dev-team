from agents.model import provider_prerequisite


def test_anthropic_prereq(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert "ANTHROPIC_API_KEY" in provider_prerequisite("anthropic:claude-sonnet-5-5")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert provider_prerequisite("anthropic:claude-sonnet-5-5") is None


def test_unknown_prefix():
    assert "unsupported" in provider_prerequisite("foo:bar")

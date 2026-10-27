"""Live smoke tests: a real model call, no fakes. Skipped without a provider; `make verify` fails on skips."""

import pytest

from agents.coder_agent import CoderAgent
from app.cli import DEMO_TASK_1, DEMO_TASK_2

pytestmark = pytest.mark.live


def test_smoke_structured_output(live_settings):
    r = CoderAgent(live_settings).run(DEMO_TASK_1)
    assert r.status == "success", r.verification_notes
    assert r.verified and r.result.executed and r.result.exit_code == 0 and not r.result.timed_out
    assert r.code.strip() and r.explanation.strip() and r.plan and all(p.strip() for p in r.plan)
    assert 1 <= r.iterations <= 10
    for path, content in r.files.items():
        assert (live_settings.workspace_dir / path).read_text() == content
        assert content in r.code
    assert "palindrome.py" in r.files and any(p.startswith("test_") for p in r.files)


def test_memory_follow_up(live_settings):
    CoderAgent(live_settings).run(DEMO_TASK_1)
    r = CoderAgent(live_settings).run(DEMO_TASK_2)  # fresh instance: proves Chroma cross-session recall
    assert r.status == "success", r.verification_notes
    assert any("palindrome" in h.text.lower() for h in r.memories_used)

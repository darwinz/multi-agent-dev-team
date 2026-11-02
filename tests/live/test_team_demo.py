"""Live team demos: the real PM and Coder on the published API task and on a requirement that forces consolidation."""

import pytest

from app.api_check import check_api, find_app
from app.team_cli import DEMO_API, DEMO_LARGE
from orchestration.project_state import MAX_TASKS
from orchestration.team_runner import TeamRunner

pytestmark = pytest.mark.live


def test_demo_api_end_to_end(live_settings):
    runner = TeamRunner(live_settings)
    runner.reset_memory()
    state = runner.run(DEMO_API)
    assert len(state["spec"]["tasks"]) >= 2
    assert state["status"] == "completed", state.get("error") or state["task_status"]
    ws = live_settings.workspace_dir
    results = check_api(find_app(ws) or "main:app", ws)
    assert results and all(r.passed for r in results), [(r.name, r.detail) for r in results if not r.passed]


def test_demo_large_consolidates(live_settings):
    runner = TeamRunner(live_settings)
    runner.reset_memory()
    state = runner.run(DEMO_LARGE)
    assert state["consolidation"] is not None
    assert len(state["tasks"]) <= MAX_TASKS
    covered = {c for t in state["tasks"] for c in t["covers"]}
    missing = [c["id"] for c in state["spec"]["acceptance_criteria"] if c["id"] not in covered]
    assert not missing, f"criteria not covered by any executed task: {missing}"

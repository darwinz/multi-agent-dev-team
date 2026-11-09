"""Live team demos: the real PM, Coder and QA on the published API task, a requirement that forces
consolidation, the linked-list grading task, and the BST grading task against the supplied fixture."""

import json
import subprocess
import sys

import pytest

from app.api_check import check_api, find_app
from app.demo_check import FIXTURE_PATH, check_bst, check_linked_list
from app.team_cli import DEMO_API, DEMO_BST, DEMO_LARGE, DEMO_LINKED_LIST
from orchestration.project_state import MAX_TASKS
from orchestration.team_runner import TeamRunner
from tools.exec_python import child_env

pytestmark = pytest.mark.live


def test_demo_api_end_to_end(live_settings):
    runner = TeamRunner(live_settings)
    runner.reset_memory()
    state = runner.run(DEMO_API)
    assert len(state["spec"]["tasks"]) >= 2
    assert state["status"] == "completed", state.get("error") or state["task_status"]
    ws = live_settings.workspace_dir
    app_spec = find_app(ws, sorted(state["code_output"]))
    assert app_spec is not None
    results = check_api(app_spec, ws)
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


def test_demo_linked_list_end_to_end(live_settings):
    runner = TeamRunner(live_settings)
    runner.reset_memory()
    final = None
    for event in runner.stream(DEMO_LINKED_LIST, demo_check=check_linked_list):
        if event.kind == "final":
            final = event.data
    assert final is not None, "the run never produced a final event"
    state = final["state"]
    assert state["status"] == "completed", state.get("error") or state["task_status"]
    demo = final["demo_check"]
    assert demo is not None and demo["passed"], demo and demo["report"]

    log = [json.loads(line) for line in (runner.run_dir / "a2a-log.jsonl").read_text().splitlines()]
    sent = [r["message"] for r in log if r["event"] == "sent"]
    requests = {m["correlation_id"] for m in sent if m["intent"] == "review_request"}
    approvals = {m["correlation_id"] for m in sent if m["intent"] == "approved"}
    assert requests & approvals, f"no review_request/approved pair shares a correlation_id: {requests} vs {approvals}"

    unreflected = [t for t, rec in state["task_status"].items() if not rec["reflected"]]
    assert state["task_status"] and not unreflected, f"tasks without a reflection: {unreflected}"


def test_bst_demo_passes_supplied_fixture(live_settings, tmp_path):
    runner = TeamRunner(live_settings)
    runner.reset_memory()
    final = None
    for event in runner.stream(DEMO_BST, demo_check=check_bst):
        if event.kind == "final":
            final = event.data
    assert final is not None, "the run never produced a final event"
    state = final["state"]
    assert state["status"] == "completed", state.get("error") or state["task_status"]
    demo = final["demo_check"]
    assert demo is not None and demo["passed"], demo and demo["report"]

    # The generated module itself, imported and run against the supplied fixture: not just report text
    ws = live_settings.workspace_dir
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--noconftest",
                          str(FIXTURE_PATH)], cwd=ws, env={**child_env(tmp_path), "PYTHONPATH": str(ws)},
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]

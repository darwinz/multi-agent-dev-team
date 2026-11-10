"""Deterministic full-pipeline E2E on the BST grading demo (spec §5.8): the real team graph, tools,
exec_python, QA's pytest run and the supplied fixture; only the PM, Coder and QA models are scripted,
so it runs offline in the default suite."""

from __future__ import annotations

import json

import pytest

from agents.config import Settings
from app.demo_check import check_bst
from app.team_cli import DEMO_BST
from tests.scenarios import bst_scripts, scripted_runner


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings.from_env(project_dir=tmp_path, exec_timeout_s=15)


def test_mocked_pipeline_builds_bst_that_passes_supplied_fixture(settings):
    """Real team graph, tools, exec_python, QA pytest run and supplied fixture; only the LLM is scripted."""
    runner = scripted_runner(settings, bst_scripts())
    final = None
    for ev in runner.stream(DEMO_BST, demo_check=check_bst):
        if ev.kind == "final":
            final = ev.data
    assert final["state"]["status"] == "completed"
    assert final["demo_check"]["passed"], final["demo_check"]["report"]
    assert "PASS supplied BST fixture" in final["demo_check"]["report"]
    assert "coverage of bst.py" in final["demo_check"]["report"]
    assert (settings.workspace_dir / "bst.py").is_file()
    assert json.loads(runner.cost_report_path.read_text())["outcome"] == "success"

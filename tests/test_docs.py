"""The README's team Mermaid diagram must match the real team graph."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from agents.pm_agent import PMAgent
from orchestration.team_graph import TeamDeps, build_team_graph
from tests.fakes import ScriptedChatModel, agent, qa_agent

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
ALIAS = {"__start__": "START", "__end__": "END"}


def _team_graph_blocks() -> list[str]:
    blocks = re.findall(r"```mermaid\n(.*?)```", README.read_text(), flags=re.S)
    return [b for b in blocks if "pm_node" in b]


def test_readme_diagram_matches_graph(settings):
    deps = TeamDeps(pm=PMAgent(settings, model=ScriptedChatModel(responses=[])), coder=agent(settings),
                    qa=qa_agent(settings))
    graph = build_team_graph(deps).get_graph()

    blocks = _team_graph_blocks()
    assert len(blocks) == 1, "README needs exactly one mermaid block describing the team graph"
    block = blocks[0]

    for node in graph.nodes:
        if node in ALIAS:
            continue
        assert re.search(rf"\b{re.escape(node)}\b", block), f"node {node} missing from README diagram"

    for e in graph.edges:
        src, dst = ALIAS.get(e.source, e.source), ALIAS.get(e.target, e.target)
        pattern = rf"\b{re.escape(src)}\b[^\n]*?-->(\|[^|\n]*\|)?\s*{re.escape(dst)}\b"
        assert re.search(pattern, block), f"edge {src} --> {dst} missing from README diagram"


def test_readme_has_no_local_notes():
    assert "LOCAL-NOTES" not in README.read_text()


def test_readme_explains_protocol():
    text = README.read_text()
    for needle in ("## Why A2A", "QA_MAX_ITERATIONS", "correlation_id", "make team-linked-list",
                   "scripted PM/Coder/QA models, real pytest subprocess execution",
                   "docs/evidence/v3/linked-list/"):
        assert needle in text, f"README is missing {needle!r}"


def test_compose_shape():
    services = yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]
    team, chroma = services["team"], services["chroma"]
    assert team["depends_on"]["chroma"]["condition"] == "service_healthy"
    assert "healthcheck" in chroma
    assert team.get("env_file"), "team must read .env via env_file"
    assert services["web"].get("profiles") and services["coder"].get("profiles")

    dockerfile = (ROOT / "Dockerfile").read_text()
    for needle in ("FROM python:3.11-slim", "-r requirements.txt", 'ENTRYPOINT ["team"]'):
        assert needle in dockerfile, f"Dockerfile is missing {needle!r}"

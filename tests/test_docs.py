"""The README's v2 Mermaid diagram must match the real team graph."""

from __future__ import annotations

import re
from pathlib import Path

from agents.config import Settings
from agents.pm_agent import PMAgent
from orchestration.team_graph import TeamDeps, build_team_graph
from tests.fakes import ScriptedChatModel, agent

README = Path(__file__).resolve().parent.parent / "README.md"
ALIAS = {"__start__": "START", "__end__": "END"}


def _team_graph_blocks() -> list[str]:
    blocks = re.findall(r"```mermaid\n(.*?)```", README.read_text(), flags=re.S)
    return [b for b in blocks if "pm_node" in b]


def test_readme_diagram_matches_graph():
    settings = Settings.from_env()
    deps = TeamDeps(pm=PMAgent(settings, model=ScriptedChatModel(responses=[])), coder=agent(settings))
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

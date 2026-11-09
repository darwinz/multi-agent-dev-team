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


def test_dockerfile_installs_project_editable():
    # A plain wheel drops tests/fixtures and web/, which the BST grading check and coder-web read from the source tree
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert re.search(r"pip install [^\n]*--no-deps -e \.", dockerfile), "the project install must be editable"


def test_build_ids_avoid_shell_readonly_names():
    # UID is readonly in bash and protected in zsh, so `UID=$(id -u) docker compose build` fails there
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "ARG APP_UID=" in dockerfile and "ARG APP_GID=" in dockerfile
    args = yaml.safe_load((ROOT / "docker-compose.yml").read_text())["x-app"]["build"]["args"]
    assert set(args) == {"APP_UID", "APP_GID"}
    command = "APP_UID=$(id -u) APP_GID=$(id -g) docker compose build"
    for doc in ("README.md", "docker-compose.yml", "docs/usage.md", "docs/architecture.md"):
        text = (ROOT / doc).read_text()
        assert not re.search(r"(?<![A-Z_])[UG]ID=\$\(id", text), f"{doc} still builds with UID/GID"
    assert command in README.read_text() and command in (ROOT / "docker-compose.yml").read_text()


def test_bind_mount_dirs_ship_with_gitkeep():
    # Pre-created so dockerd doesn't create them root-owned on Linux (the container user couldn't write)
    for d in ("workspace", ".coder", "docs/cost_reports"):
        assert (ROOT / d / ".gitkeep").is_file(), f"{d}/.gitkeep is missing"


V4_HEADINGS = ("## Production v4", "### Quickstart (Docker)", "### Quickstart (local, 1Password)", "### Configuration",
               "### Callable interface", "### Running the tests", "### Traces, logs and cost reports",
               "### Reading the outcome", "### Test map")
EVIDENCE_MAP = ROOT / "docs" / "evidence" / "v4" / "README.md"


def test_readme_v4_sections():
    text = README.read_text()
    for heading in V4_HEADINGS:
        assert re.search(rf"^{re.escape(heading)}\s*$", text, flags=re.M), f"README is missing {heading!r}"


def test_readme_env_vars_match_settings():
    from agents.config import _ENV_VARS

    text = README.read_text()
    for var in _ENV_VARS.values():
        assert f"`{var}`" in text, f"README configuration table is missing {var}"


def test_readme_demo_flags_match_cli():
    from app.team_cli import DEMOS

    text = README.read_text()
    for demo in DEMOS:
        assert f"--demo {demo}" in text, f"README is missing --demo {demo}"


def test_evidence_map_covers_c4():
    text = EVIDENCE_MAP.read_text()
    for n in range(1, 10):
        assert f"C4.{n}" in text, f"evidence map is missing C4.{n}"
    links = [m for m in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", text) if "://" not in m]
    assert links, "evidence map has no relative links"
    for link in links:
        assert (EVIDENCE_MAP.parent / link).exists(), f"evidence map link does not resolve: {link}"


def test_readme_v4_diagram_block_is_separate():
    blocks = re.findall(r"```mermaid\n(.*?)```", README.read_text(), flags=re.S)
    assert any("breaker" in b.lower() and "cost" in b.lower() for b in blocks if "pm_node" not in b)

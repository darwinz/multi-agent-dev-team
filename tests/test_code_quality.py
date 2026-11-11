"""Code-quality rules the grading rubric checks: no function longer than 50 lines, anywhere in the repo."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ["agents", "app", "memory", "observability", "orchestration", "resilience", "tools", "scripts", "tests"]
MAX_LINES = 50


def _functions() -> list[tuple[str, int]]:
    """(location, length in lines) for every function and method in the project's Python files."""
    found = []
    for pkg in PACKAGES:
        for path in sorted((ROOT / pkg).rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    where = f"{path.relative_to(ROOT)}:{node.lineno} {node.name}"
                    found.append((where, node.end_lineno - node.lineno + 1))
    return found


def test_scan_finds_the_project_functions():
    names = {where.split()[-1] for where, _ in _functions()}
    assert {"build_team_graph", "build_graph", "exec_python", "main"} <= names


def test_no_function_longer_than_50_lines():
    too_long = [f"{where} ({n} lines)" for where, n in _functions() if n > MAX_LINES]
    assert not too_long, "functions over 50 lines:\n" + "\n".join(too_long)

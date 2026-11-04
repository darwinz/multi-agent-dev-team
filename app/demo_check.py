"""The linked-list grading demo's check (spec §11): judged from the run's final state, never the workspace."""

from __future__ import annotations

import ast
from pathlib import Path

from orchestration.project_state import state_to_jsonable

METHODS = ("insert", "delete", "search", "reverse")


def _line(ok: bool, what: str, detail: str) -> str:
    return f"{'PASS' if ok else 'FAIL'} {what}: {detail}"


def _approved_line(state: dict) -> tuple[bool, str]:
    what = "every task approved with QA tests run"
    tasks = [t["task_id"] for t in state.get("tasks") or []]
    if not tasks:
        return False, _line(False, what, "the run has no tasks")
    records = state.get("task_status") or {}
    reports = state.get("qa_reports") or []
    good, bad = [], []
    for task_id in tasks:
        rec = records.get(task_id, {})
        last = next((r for r in reversed(reports) if r["task_id"] == task_id), None)
        if rec.get("status") != "done" or rec.get("qa_outcome") != "approved":
            bad.append(f"{task_id} {rec.get('status', 'pending')} ({rec.get('qa_outcome') or 'no QA outcome'})")
        elif last is None or not last["approved"] or last["tests_run"] < 1:
            bad.append(f"{task_id} has no approved QA round with tests run")
        else:
            good.append(f"{task_id} ({last['tests_run']} tests, round {last['round']})")
    return not bad, _line(not bad, what, "; ".join(bad or good))


def _called_methods(source: str) -> dict[str, set[str]]:
    """{test function: attribute names it calls}, following calls to the module's own helper functions."""
    funcs = {n.name: n for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    attrs: dict[str, set[str]] = {}
    helpers: dict[str, set[str]] = {}
    for name, fn in funcs.items():
        calls = [n.func for n in ast.walk(fn) if isinstance(n, ast.Call)]
        attrs[name] = {c.attr for c in calls if isinstance(c, ast.Attribute)}
        helpers[name] = {c.id for c in calls if isinstance(c, ast.Name) and c.id in funcs}
    out: dict[str, set[str]] = {}
    for name in funcs:
        if not name.startswith("test_"):
            continue
        seen, todo = {name}, [name]
        while todo:
            for h in helpers[todo.pop()] - seen:
                seen.add(h)
                todo.append(h)
        out[name] = set().union(*(attrs[f] for f in seen))
    return out


def _methods_line(state: dict) -> tuple[bool, str]:
    what = f"{', '.join(METHODS)} each called in a canonical QA test"
    suites = state.get("approved_suites") or {}
    where: dict[str, list[str]] = {m: [] for m in METHODS}
    for task_id, suite in sorted(suites.items()):
        try:
            called = _called_methods(suite["content"])
        except SyntaxError:
            continue
        for test, names in called.items():
            for m in METHODS:
                if m in names:
                    where[m].append(f"{task_id}::{test}")
    missing = [m for m in METHODS if not where[m]]
    if missing:
        source = f"in {', '.join(sorted(suites))}" if suites else "(no approved suites)"
        return False, _line(False, what, f"never called {source}: {', '.join(missing)}")
    return True, _line(True, what, "; ".join(f"{m} ({where[m][0]})" for m in METHODS))


def _regression_line(state: dict) -> tuple[bool, str]:
    what = "project regression check"
    reg = state.get("regression")
    if not reg:
        return False, _line(False, what, "no regression run")
    expected = set(state.get("approved_suites") or {})
    not_run = sorted(expected - set(reg["suites_run"]))
    problems = []
    if not_run:
        problems.append(f"suites not restored and run: {', '.join(not_run)}")
    if not reg["approved"] or reg["skipped"] or reg["missing"]:
        problems.append(f"not approved ({reg['passed']}/{reg['tests_run']} passed, {reg['skipped']} skipped, "
                        f"{len(reg['missing'])} missing)")
    if problems:
        return False, _line(False, what, "; ".join(problems))
    return True, _line(True, what, f"{reg['passed']}/{reg['tests_run']} tests passed over "
                                   f"{', '.join(sorted(reg['suites_run']))}")


def check_linked_list(workspace: Path, state: dict) -> tuple[bool, str]:
    """One PASS/FAIL line per check. Judges the canonical suites and reports in `state` (model objects or
    their JSON dumps); it never reads or re-runs the workspace's test_qa_*.py copies, which the Coder
    could have edited. `workspace` is unused, kept for the runner's demo_check signature."""
    state = state_to_jsonable(state)
    results = [_approved_line(state), _methods_line(state), _regression_line(state)]
    return all(ok for ok, _ in results), "\n".join(line for _, line in results)

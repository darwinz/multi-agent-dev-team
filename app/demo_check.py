"""The grading demos' checks. linked-list (spec §11): judged from the run's final state, never the
workspace. bst (spec §5.8): the supplied fixture run against the generated module, plus QA's coverage."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from orchestration.handoff import qa_test_file
from orchestration.project_state import state_to_jsonable
from tools.exec_python import child_env
from tools.junit import parse_junit

METHODS = ("insert", "delete", "search", "reverse")

# The BST demo's grading contract, kept with the tests (spec §5.8)
FIXTURE_PATH = Path(__file__).parents[1] / "tests/fixtures/bst_fixture.py"
FIXTURE_TIMEOUT_S = 60
_TAIL_LINES = 8


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


def _tail(text: str) -> str:
    """The last few non-empty output lines on one line, so the report stays one line per check."""
    text = re.sub(r"\S*bst_fixture\.py", "bst_fixture.py", text)  # node ids without the long relative path
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("=")]
    tail = " | ".join(lines[-_TAIL_LINES:]) or "no output"
    return tail if len(tail) <= 600 else "..." + tail[-597:]


def _run(args: list[str], cwd: Path, workspace: Path, tmp: Path, **env: str) -> subprocess.CompletedProcess:
    """A pytest subprocess on the generated module: allowlisted env (no secrets), workspace importable."""
    return subprocess.run(args, cwd=cwd, env={**child_env(tmp), "PYTHONPATH": str(workspace), **env},
                          capture_output=True, text=True, timeout=FIXTURE_TIMEOUT_S, check=False)


def _fixture_tests() -> set[str]:
    """The supplied fixture's test function names, read from its source."""
    tree = ast.parse(FIXTURE_PATH.read_text(encoding="utf-8"))
    return {n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")}


def _fixture_line(workspace: Path, tmp: Path) -> tuple[bool, str]:
    what = "supplied BST fixture"
    if not workspace.is_dir():
        return False, _line(False, what, f"no workspace at {workspace}")
    junit = tmp / "fixture.xml"
    # --noconftest: neither a workspace conftest.py nor the repo's test conftest can reshape the run
    args = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--noconftest",
            f"--junitxml={junit}", str(FIXTURE_PATH)]
    try:
        out = _run(args, workspace, workspace, tmp)
    except subprocess.TimeoutExpired:
        return False, _line(False, what, f"timed out after {FIXTURE_TIMEOUT_S} s")
    tail = _tail(out.stdout + out.stderr)
    if out.returncode != 0:
        return False, _line(False, what, f"exit {out.returncode}: {tail}")
    # Exit 0 alone is not proof: a workspace pytest.py or a module the fixture imports can shadow the real
    # one, or the code can end the process early. Only pytest's own result file, with every fixture test
    # passed, counts.
    result = parse_junit(junit)
    if result is None:
        return False, _line(False, what, f"pytest exited 0 but no result file (did the fixture run?): {tail}")
    expected = _fixture_tests()
    ran = {c.name for c in result.cases}
    passed = {c.name for c in result.cases if c.outcome == "passed"}
    if (len(result.cases) != len(expected) or ran != expected or passed != expected
            or result.failures or result.errors or result.skipped):
        return False, _line(False, what, f"pytest exited 0 but the result file shows {len(passed)}/{len(expected)} "
                                         f"fixture tests passed ({len(result.cases)} cases, {result.failures} failed, "
                                         f"{result.errors} errors, {result.skipped} skipped): {tail}")
    return True, _line(True, what, f"{len(passed)}/{len(expected)} fixture tests passed: {tail}")


def _coverage(workspace: Path, state: dict, tmp: Path) -> str:
    """bst.py's line coverage under the run's canonical QA suites (the approved ones in `state`, never the
    workspace copies), e.g. "87%", or "n/a (<reason>)"."""
    suites = state.get("approved_suites") or {}
    if not suites:
        return "n/a (no approved QA suites)"
    if not (workspace / "bst.py").is_file():
        return "n/a (no bst.py in the workspace)"
    run_dir = tmp / "coverage"
    run_dir.mkdir()
    files = []
    for task_id, suite in sorted(suites.items()):
        files.append(run_dir / qa_test_file(task_id))
        files[-1].write_text(suite["content"], encoding="utf-8")
    ini = run_dir / "pytest.ini"
    ini.write_text("[pytest]\n", encoding="utf-8")
    args = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--noconftest", "-c", str(ini),
            f"--rootdir={run_dir}", "--cov=bst", "--cov-report=term", *map(str, files)]
    try:
        out = _run(args, run_dir, workspace, tmp, COVERAGE_FILE=str(run_dir / ".coverage"))
    except subprocess.TimeoutExpired:
        return f"n/a (QA's suites timed out after {FIXTURE_TIMEOUT_S} s)"
    for line in out.stdout.splitlines():
        cols = line.split()
        if cols and Path(cols[0]).name == "bst.py" and cols[-1].endswith("%"):
            return cols[-1] if out.returncode == 0 else f"{cols[-1]} (QA's suites exited {out.returncode})"
    return f"n/a (no bst.py line in the coverage report; pytest exit {out.returncode})"


def check_bst(workspace: Path, state: dict) -> tuple[bool, str]:
    """The BST demo: every task approved with QA tests run, and the supplied fixture passes against the
    workspace's bst.py. A third line reports bst.py's coverage under QA's tests; it is never gated."""
    state = state_to_jsonable(state)
    workspace = Path(workspace)
    with tempfile.TemporaryDirectory(prefix="bst-check-") as name:
        tmp = Path(name)
        results = [_approved_line(state), _fixture_line(workspace, tmp)]
        try:
            coverage = _coverage(workspace, state, tmp)
        except Exception as e:  # noqa: BLE001 - reported, not gated: a measurement problem never fails or crashes the check
            coverage = f"n/a ({type(e).__name__}: {e})"
    lines = [line for _, line in results] + [f"coverage of bst.py under QA's tests: {coverage} (reported, not gated)"]
    return all(ok for ok, _ in results), "\n".join(lines)

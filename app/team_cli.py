"""`team`: run the PM -> Coder <-> QA team from the terminal."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError
from rich.console import Console
from rich.padding import Padding
from rich.text import Text

from agents.config import Settings
from agents.events import StepEvent, TeamEvent
from agents.model import provider_prerequisite
from app.api_check import check_api, find_app
from app.demo_check import check_bst, check_linked_list
from app.render import (
    render_a2a,
    render_consolidation,
    render_event,
    render_qa_report,
    render_reflection,
    render_spec,
    render_task_header,
    render_team_summary,
)
from memory.semantic import MemoryUnavailableError
from observability.logs import configure_logging
from orchestration.team_runner import TeamRunner
from tools.workspace import WorkspaceBusyError

DEMO_API = (
    "Build a REST API with two endpoints: GET /health (returns {status: ok}) and POST /echo "
    "(returns the JSON body it receives). Include basic input validation. "
    "Use FastAPI; invalid input to /echo (malformed JSON, a non-JSON-object body, or no body) "
    "must return HTTP 422."
)

DEMO_LARGE = (
    "Build a tiny in-memory to-do library in Python with these separately testable features, each with "
    "pytest tests: (1) add a to-do item; (2) list all items; (3) mark an item complete; (4) delete an item; "
    "(5) rename an item; (6) set a due date on an item; (7) set a priority (high, medium or low) on an item; "
    "(8) filter items by status (open or complete); (9) search items by text; (10) export all items to JSON; "
    "(11) import items from JSON. Validate inputs and raise clear errors for unknown item ids."
)

# The Module 3 grading demo, verbatim (spec §2)
DEMO_LINKED_LIST = (
    "Implement a Python module that contains a linked list class with insert, delete, search, and reverse "
    "methods. All methods must be covered by tests."
)

# The C4.7 grading demo: the published sentence verbatim, then the interface the supplied fixture
# (tests/fixtures/bst_fixture.py) checks (spec §5.8)
DEMO_BST = (
    "Build a Python module that implements a binary search tree with insert, search, and in-order traversal "
    "methods, with full test coverage."
    "\n\nInterface (the supplied fixture's contract): "
    "module bst.py at the workspace root; class BinarySearchTree with insert(value) -> None, "
    "search(value) -> bool, and inorder() -> list returning the values in ascending order; "
    "duplicate inserts are ignored; values are mutually comparable."
)

DEMOS = {"api": DEMO_API, "large": DEMO_LARGE, "linked-list": DEMO_LINKED_LIST, "bst": DEMO_BST}
DEMO_CHECKS = {"linked-list": ("Linked-list check", check_linked_list), "bst": ("BST fixture check", check_bst)}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="team", description="Run the PM -> Coder <-> QA team on a requirement.")
    p.add_argument("requirement", nargs="?", help="the free-text requirement")
    p.add_argument("--demo", choices=list(DEMOS), help="api: the Module 2 grading task; "
                   "large: a requirement that yields more than 8 tasks (exercises consolidation); "
                   "linked-list: the Module 3 grading task, then the linked-list check; "
                   "bst: the binary search tree grading task, then the supplied BST fixture")
    p.add_argument("--model", help="Coder model, provider-prefixed, e.g. anthropic:claude-haiku-5-5 or openai:gpt-4o")
    p.add_argument("--pm-model", help="PM model, provider-prefixed (default anthropic:claude-sonnet-5-5)")
    p.add_argument("--qa-model", help="QA model, provider-prefixed (default anthropic:claude-haiku-5-5)")
    p.add_argument("--max-iterations", type=int, help="per-task model-call limit, 1 to 10 (default 10)")
    p.add_argument("--timeout", type=float, help="exec_python timeout in seconds (default 10)")
    # A string, so a bad value gets the same "invalid setting" report from Settings as the env var
    p.add_argument("--qa-max-iterations", metavar="N",
                   help="QA test rounds per task, 1 to 5 (default 5, or QA_MAX_ITERATIONS)")
    p.add_argument("--reset-memory", action="store_true", help="clear both memory layers and the workspace first")
    p.add_argument("--json", action="store_true", help="print only the final ProjectState JSON on stdout")
    return p


def _api_check(ws: Path, written: list[str]) -> tuple[bool, str]:
    app_spec = find_app(ws, written)
    if app_spec is None:
        return False, f"FAIL find app: no file this run delivered assigns app = FastAPI(...) in {ws}"
    results = check_api(app_spec, ws)
    report = "\n".join(f"{'PASS' if r.passed else 'FAIL'} {r.name}" + (f": {r.detail}" if r.detail else "")
                       for r in results)
    return bool(results) and all(r.passed for r in results), report


def _usage_error(parser: argparse.ArgumentParser, args: argparse.Namespace) -> bool:
    """Report a missing or conflicting requirement/--demo; True if the arguments can't run."""
    if not (args.requirement or args.demo or args.reset_memory):
        parser.print_usage(sys.stderr)
        print("team: error: give a requirement, --demo, or --reset-memory", file=sys.stderr)
        return True
    if args.requirement and args.demo:
        print("team: error: give either a requirement or --demo, not both", file=sys.stderr)
        return True
    return False


def _make_runner(args: argparse.Namespace, requirement: str | None, err: Console) -> TeamRunner | None:
    """Settings from flags and env, a provider check per model, then the runner; None (reported) on a config error."""
    try:
        settings = Settings.from_env(model=args.model, pm_model=args.pm_model, qa_model=args.qa_model,
                                     max_iterations=args.max_iterations,
                                     exec_timeout_s=args.timeout, qa_max_iterations=args.qa_max_iterations)
    except ValidationError as e:
        for problem in e.errors():
            err.print(f"team: invalid setting {'.'.join(map(str, problem['loc']))}: {problem['msg']}", markup=False)
        return None
    if requirement:
        for model in dict.fromkeys(settings.model_for(r) for r in ("pm", "coder", "qa")):
            if reason := provider_prerequisite(model):
                err.print(f"team: cannot run: {reason}", markup=False)
                return None
    configure_logging(settings.log_level)  # first: TeamRunner's startup (Chroma retries) already logs
    try:
        return TeamRunner(settings)
    except MemoryUnavailableError as e:  # CHROMA_HOST set but the server is down: say so, no traceback
        err.print(f"team: cannot run: {e}", markup=False)
        return None


_RENDERERS = {
    "spec": render_spec,
    "consolidation": render_consolidation,
    "handoff": render_task_header,
    "reflection": render_reflection,
    "qa_report": render_qa_report,
    "a2a": render_a2a,
}


def _render_final(console: Console, data: dict, check_title: str | None) -> None:
    """The run summary, then each check that ran with its PASSED/FAILED verdict and report."""
    render_team_summary(console, data["state"])
    for title, check in (("API fixture check", data.get("api_check")), (check_title, data.get("demo_check"))):
        if check is not None:
            console.print(Text(f"{title}: " + ("PASSED" if check["passed"] else "FAILED"),
                               style="bold green" if check["passed"] else "bold red"))
            console.print(Text(check["report"]), soft_wrap=True)


def _render(console: Console, event: TeamEvent, check_title: str | None) -> None:
    """Print one team event to the console."""
    k = event.kind
    if k in _RENDERERS:
        _RENDERERS[k](console, event.data)
    elif k == "start":
        console.print(Text(f"run dir: {event.data['run_dir']}", style="dim"), soft_wrap=True)
    elif k == "coder":  # the Coder's own step events, indented under the task
        with console.capture() as cap:
            render_event(console, StepEvent.model_validate(event.data))
        console.print(Padding(Text.from_ansi(cap.get().rstrip("\n")), (0, 0, 0, 4)))
    elif k == "task_done":
        d = event.data
        console.print(Text(f"{event.task_id}: {d['status']} (QA: {d.get('qa_outcome')} after "
                           f"{d.get('qa_rounds', 0)} round(s); Coder: {d.get('coder_status')}, "
                           f"{d.get('iterations', 0)} iterations)", style="bold"), soft_wrap=True)
    elif k == "final":
        _render_final(console, event.data, check_title)


def _exit_code(demo: str | None, final: dict, data: dict) -> int:
    """0 only for a completed run whose demo check (if the demo has one) ran and passed; otherwise 1."""
    ok = final["status"] == "completed"
    if demo == "api" or demo in DEMO_CHECKS:  # a demo's exit also needs its check to have run and passed
        check = data.get("api_check") if demo == "api" else data.get("demo_check")
        ok = ok and check is not None and check["passed"]
    return 0 if ok else 1


def main(argv: list[str] | None = None, *, runner: TeamRunner | None = None) -> int:
    """Run the team from the command line; returns the process exit code (0, 1, 2 or 130)."""
    parser = _parser()
    args = parser.parse_args(argv)
    if _usage_error(parser, args):
        return 2
    console = Console(stderr=args.json)
    err = Console(stderr=True)
    requirement = DEMOS[args.demo] if args.demo else args.requirement
    if runner is not None:
        configure_logging(runner.settings.log_level)
    elif (runner := _make_runner(args, requirement, err)) is None:  # configures logging itself
        return 2
    final: dict = {}
    try:
        if args.reset_memory:
            runner.reset_memory()
            err.print("memory and workspace cleared", style="dim")
        if not requirement:
            return 0
        check_title, check_fn = DEMO_CHECKS.get(args.demo, (None, None))
        for event in runner.stream(requirement, api_check=_api_check if args.demo == "api" else None,
                                   demo_check=check_fn):
            _render(console, event, check_title)
            if event.kind == "final":
                final = event.data
        # The report is written as the stream ends, after the final event
        path = runner.cost_report_path
        console.print(Text(f"cost report: {path if path else 'unavailable (see events.jsonl)'}", style="dim"),
                      soft_wrap=True)
    except WorkspaceBusyError as e:
        err.print(f"team: {e}", markup=False)
        return 2
    except KeyboardInterrupt:
        err.print("team: interrupted (the workspace lock has been released)", markup=False)
        return 130
    assert final, "the team stream ended without a final event"
    if args.json:
        print(json.dumps(final["state"], indent=2))
    return _exit_code(args.demo, final["state"], final)


if __name__ == "__main__":
    sys.exit(main())

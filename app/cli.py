"""`coder`: run the Coder agent from the terminal."""

from __future__ import annotations

import argparse
import json
import sys

from pydantic import ValidationError
from rich.console import Console

from agents.coder_agent import CoderAgent
from agents.config import Settings
from agents.model import provider_prerequisite
from agents.result import CoderResult
from app.render import render_event, render_result
from memory.semantic import MemoryUnavailableError
from tools.workspace import WorkspaceBusyError

DEMO_TASK_1 = (
    "Create palindrome.py with a function is_palindrome(text: str) -> bool that ignores case and spaces, "
    "plus test_palindrome.py with pytest tests covering at least four cases. Run the tests."
)
DEMO_TASK_2 = "Now make the palindrome checker also ignore punctuation, add tests for that, and run all the tests."


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="coder", description="Run the Coder agent on a natural-language coding task.")
    p.add_argument("task", nargs="?", help="the coding task")
    p.add_argument("--model", help="provider-prefixed model, e.g. anthropic:claude-sonnet-5-5 or openai:gpt-4o")
    p.add_argument("--max-iterations", type=int, help="model-call limit, 1 to 10 (default 10)")
    p.add_argument("--timeout", type=float, help="exec_python timeout in seconds (default 10)")
    p.add_argument("--reset-memory", action="store_true", help="clear both memory layers and the workspace first")
    p.add_argument("--json", action="store_true", help="print only the CoderResult JSON on stdout")
    p.add_argument("--demo", action="store_true", help="run the two-task palindrome demo (memory recall)")
    return p


def _make_agent(args: argparse.Namespace, tasks: list[str], err: Console) -> CoderAgent | None:
    """Settings from flags and env, a provider check, then the agent; None (reported) on a config error."""
    try:
        settings = Settings.from_env(model=args.model, max_iterations=args.max_iterations,
                                     exec_timeout_s=args.timeout)
    except ValidationError as e:
        for problem in e.errors():
            err.print(f"coder: invalid setting {'.'.join(map(str, problem['loc']))}: {problem['msg']}", markup=False)
        return None
    if tasks and (reason := provider_prerequisite(settings.model)):
        err.print(f"coder: cannot run: {reason}", markup=False)
        return None
    try:
        return CoderAgent(settings)
    except MemoryUnavailableError as e:  # CHROMA_HOST set but the server is down: say so, no traceback
        err.print(f"coder: cannot run: {e}", markup=False)
        return None


def _run_task(agent: CoderAgent, task: str, console: Console) -> CoderResult:
    """Stream one Coder run, rendering each step; returns the validated result."""
    result = None
    for event in agent.stream(task):
        if event.kind == "result":
            result = CoderResult.model_validate(event.data["result"])
        else:
            render_event(console, event)
    assert result is not None
    return result


def _print_json(results: list[CoderResult]) -> None:
    """One CoderResult as JSON, or a JSON list for the two-task demo."""
    if len(results) == 1:
        print(results[0].model_dump_json(indent=2))
    else:
        print(json.dumps([r.model_dump(mode="json") for r in results], indent=2))


def main(argv: list[str] | None = None, *, agent: CoderAgent | None = None) -> int:
    """Run the Coder from the command line; returns the process exit code (0, 1, 2 or 130)."""
    parser = _parser()
    args = parser.parse_args(argv)
    if not (args.task or args.demo or args.reset_memory):
        parser.print_usage(sys.stderr)
        print("coder: error: give a task, --demo, or --reset-memory", file=sys.stderr)
        return 2
    if args.task and args.demo:
        print("coder: error: give either a task or --demo, not both", file=sys.stderr)
        return 2
    console = Console(stderr=args.json)
    err = Console(stderr=True)
    tasks = [DEMO_TASK_1, DEMO_TASK_2] if args.demo else [args.task] if args.task else []
    if agent is None and (agent := _make_agent(args, tasks, err)) is None:
        return 2
    results: list[CoderResult] = []
    try:
        if args.reset_memory:
            agent.reset_memory()
            err.print("memory and workspace cleared", style="dim")
        for i, task in enumerate(tasks):
            if i:
                console.rule("Follow-up: memory recall")
            results.append(_run_task(agent, task, console))
            if not args.json:
                render_result(console, results[-1])
    except WorkspaceBusyError as e:
        err.print(f"coder: {e}", markup=False)
        return 2
    except KeyboardInterrupt:
        err.print("coder: interrupted (the workspace lock has been released)", markup=False)
        return 130
    if args.json and results:
        _print_json(results)
    return 0 if all(r.status == "success" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

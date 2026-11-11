"""TeamRunner: runs the team graph under one workspace lock and writes evidence files (spec §11)."""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, get_args
from uuid import uuid4

from agents.coder_agent import CoderAgent
from agents.config import Role, Settings
from agents.events import TeamEvent
from agents.pm_agent import PMAgent
from agents.qa_agent import QAAgent, canonical_suite_path
from observability.context import RunContext, run_scope
from observability.costs import build_cost_report, write_cost_report
from observability.logs import log_event, redact, run_log_file
from observability.tracing import resolve_trace_url, root_trace
from orchestration.a2a import CODER_ACCEPTS, QA_ACCEPTS, Broker
from orchestration.project_state import initial_project_state, state_to_jsonable
from orchestration.team_graph import TeamDeps, build_team_graph, team_recursion_limit
from tools.workspace import RunLock

ApiCheck = Callable[[Path, list[str]], tuple[bool, str]]  # (workspace, files this run wrote)
DemoCheck = Callable[[Path, dict], tuple[bool, str]]  # (workspace, final state as JSON)

A2A_LOG = "a2a-log.jsonl"
ROLES: tuple[Role, ...] = get_args(Role)
OUTCOME = {"completed": "success", "partial": "unresolved", "failed": "failed"}  # final status → report outcome


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _capabilities(agent: str, accepts: frozenset[tuple[str, str]]) -> dict:
    return {"event": "capabilities", "agent": agent,
            "accepts": [{"sender": s, "intent": i} for s, i in sorted(accepts)]}


@dataclass
class _Run:
    """Bookkeeping for one run, shared by `TeamRunner.stream` and its helpers (its `finally` reads it)."""

    run_id: str
    run_dir: Path
    ctx: RunContext
    started: str
    requirement: str
    values: dict = field(default_factory=dict)  # the graph's latest full state
    outcome: str | None = None
    error: dict | None = None
    trace: Any = None  # the root run tree while tracing; its URL is resolved once, at run end
    trace_resolved: bool = False


class TeamRunner:
    """Runs the team graph under one workspace lock and writes the run's evidence files (spec §11)."""

    def __init__(self, settings: Settings, *, pm: PMAgent | None = None, coder: CoderAgent | None = None,
                 qa: QAAgent | None = None, broker_factory: Callable[[], Broker] | None = None) -> None:
        """Use the given agents (tests inject fakes) or build each from `settings`."""
        self.settings = settings
        self.broker_factory = broker_factory or Broker  # one fresh broker per run
        self.pm = pm or PMAgent(settings)
        self.coder = coder or CoderAgent(settings)
        self.qa = qa or QAAgent(settings)
        self.run_dir: Path | None = None
        self.final_state: dict | None = None
        self.cost_report_path: Path | None = None

    def reset_memory(self) -> None:
        """Clear both memory layers and the workspace."""
        self.coder.reset_memory()  # takes the RunLock itself: never call while stream() is running

    def run(self, requirement: str) -> dict:
        """Run the team to completion and return the final state (the non-streaming form of `stream`)."""
        for _ in self.stream(requirement):
            pass
        assert self.final_state is not None
        return self.final_state

    def stream(self, requirement: str, *, api_check: ApiCheck | None = None,
               demo_check: DemoCheck | None = None) -> Iterator[TeamEvent]:
        """Run the team on `requirement`, yielding `TeamEvent`s; a cost report is written however the run ends."""
        self.run_dir = None
        self.final_state = None
        self.cost_report_path = None
        # Lock first: a busy workspace must leave no run directory (and no cost report) behind
        with RunLock(self.settings.state_dir):
            run = self._new_run(requirement)
            # run_log_file inside run_scope, so events.jsonl keeps only this run's records
            with run_scope(run.ctx), run_log_file(run.run_dir / "events.jsonl"):
                try:
                    log_event("run_start", requirement_chars=len(requirement),
                              models={r: self.settings.model_for(r) for r in ROLES})
                    with root_trace(run.run_id, requirement) as run.trace:
                        yield from self._traced_run(run, api_check, demo_check)
                except BaseException as e:  # incl. GeneratorExit (closed early) and KeyboardInterrupt
                    self._on_abort(run, e)
                    raise
                finally:
                    self._finish(run)

    def _new_run(self, requirement: str) -> _Run:
        """A fresh run id and its run directory under the state dir."""
        run_id = uuid4().hex[:12]
        run_dir = self.settings.state_dir / "runs" / run_id
        run_dir.mkdir(parents=True)
        self.run_dir = run_dir
        return _Run(run_id=run_id, run_dir=run_dir, ctx=RunContext(run_id=run_id), started=_now(),
                    requirement=requirement)

    def _traced_run(self, run: _Run, api_check: ApiCheck | None,
                    demo_check: DemoCheck | None) -> Iterator[TeamEvent]:
        """Everything inside the root trace: the graph, the trace URL, the checks and the final state."""
        (run.run_dir / "requirement.txt").write_text(run.requirement)
        # The capability advertisements open the A2A log (spec §5)
        self._append_a2a(_capabilities("coder", CODER_ACCEPTS))
        self._append_a2a(_capabilities("qa", QA_ACCEPTS))
        yield TeamEvent(kind="start", data={"run_id": run.run_id, "run_dir": str(run.run_dir)})
        yield from self._stream_graph(run)
        state = state_to_jsonable(run.values)
        # Inside the root, after the graph: the root run is posted by now, and the wait is bounded
        run.trace_resolved = True
        self._resolve_trace(run.ctx, run.trace)
        # Still under the lock, so the checks and the evidence describe the same workspace
        api, demo = self._run_checks(state, api_check, demo_check)
        run.outcome = OUTCOME[state["status"]]
        self._write_json("final-state.json", state)
        self.final_state = state
        # outcome is already set: a consumer that closes the generator after `final` is no abort
        yield TeamEvent(kind="final", data={"state": state, "api_check": api, "demo_check": demo})

    def _stream_graph(self, run: _Run) -> Iterator[TeamEvent]:
        """Stream the team graph, turning its chunks into events and evidence files; keeps `run.values` current."""
        s = self.settings
        deps = TeamDeps(pm=self.pm, coder=self.coder, qa=self.qa, broker=self.broker_factory(),
                        qa_max_iterations=s.qa_max_iterations)
        pending: list[str] = []  # snapshot files waiting for the step's post-update values
        for chunk in build_team_graph(deps).stream(
            initial_project_state(run.requirement),
            stream_mode=["updates", "custom", "values"],
            version="v2",
            config={"recursion_limit": team_recursion_limit(s.qa_max_iterations)},
        ):
            kind, data = chunk["type"], chunk["data"]
            if kind == "values":
                run.values = data
                for name in pending:
                    self._write_json(name, state_to_jsonable(run.values))
                pending.clear()
            elif kind == "custom":
                event = TeamEvent.model_validate(data)
                self._on_custom(event)
                yield event
            elif kind == "updates":
                for node, update in data.items():
                    yield from self._on_update(node, update or {}, pending)
        for name in pending:  # defensive: the graph always ends with a values chunk
            self._write_json(name, state_to_jsonable(run.values))

    def _run_checks(self, state: dict, api_check: ApiCheck | None,
                    demo_check: DemoCheck | None) -> tuple[dict | None, dict | None]:
        """Run the optional API and demo checks on the final workspace (skipped when the run failed)."""
        api: dict | None = None
        demo: dict | None = None
        if state["status"] == "failed":
            return api, demo
        assert self.run_dir is not None
        workspace = self.settings.workspace_dir
        if api_check is not None:
            passed, report = api_check(workspace, sorted(state["code_output"]))
            (self.run_dir / "api-check.txt").write_text(report)
            api = {"passed": passed, "report": report}
        if demo_check is not None:
            passed, report = demo_check(workspace, state)
            (self.run_dir / "demo-check.txt").write_text(report)
            demo = {"passed": passed, "report": report}
        return api, demo

    def _on_abort(self, run: _Run, e: BaseException) -> None:
        """Record an abort (unless the run already had an outcome) and save whatever state there is."""
        if run.outcome is not None:
            return
        run.outcome = "aborted"
        run.error = {"type": type(e).__name__, "message": redact(str(e))}
        log_event("run_aborted", level=logging.ERROR, **run.error)
        self._write_json("final-state.json", state_to_jsonable(run.values or initial_project_state(run.requirement)))

    def _finish(self, run: _Run) -> None:
        """Always runs: resolve the trace URL if not done yet, then write the cost report (never raising)."""
        s = self.settings
        if run.trace is not None and not run.trace_resolved:  # aborted: still try, bounded, never raising
            with contextlib.suppress(Exception):
                self._resolve_trace(run.ctx, run.trace)
        report = build_cost_report(
            run_id=run.run_id, started_at=run.started, finished_at=_now(), requirement=run.requirement,
            outcome=run.outcome or "aborted", models={r: s.model_for(r) for r in ROLES},
            calls=list(run.ctx.calls), trace_url=run.ctx.trace_url, error=run.error)
        written, errors = write_cost_report(report, s.cost_reports_dir, run.run_dir)
        for e in errors:  # logged, never raised: the run's own outcome or exception proceeds
            log_event("cost_report_failed", level=logging.ERROR, type=type(e).__name__, message=redact(str(e)))
        docs_copy, run_copy = s.cost_reports_dir / f"{run.run_id}.json", run.run_dir / "cost-report.json"
        self.cost_report_path = next((p for p in (docs_copy, run_copy) if p in written), None)
        log_event("run_end", outcome=report["outcome"], cost_usd=report["total"]["cost_usd"],
                  report=str(self.cost_report_path))

    @staticmethod
    def _resolve_trace(ctx: RunContext, trace) -> None:
        ctx.trace_url = resolve_trace_url(trace)
        if ctx.trace_url:
            log_event("trace", url=ctx.trace_url)

    def _on_custom(self, event: TeamEvent) -> None:
        """Evidence from in-node events: A2A records, reflections and QA rounds."""
        assert self.run_dir is not None
        if event.kind == "a2a":
            self._append_a2a(event.data)
        elif event.kind == "reflection":
            self._write_json(f"reflection-{event.task_id}.json", event.data)
        elif event.kind == "qa_report":
            self._write_json(f"qa-{event.task_id}-r{event.data['round']}.json", event.data)
            if event.data["suite_sha"]:  # a suite ran this round, so its canonical copy is this run's
                tests = canonical_suite_path(self.settings, event.task_id).read_text(encoding="utf-8")
                (self.run_dir / f"qa-tests-{event.task_id}.py").write_text(tests, encoding="utf-8")

    def _on_update(self, node: str, update: dict, pending: list[str]) -> Iterator[TeamEvent]:
        if node == "pm_node":
            spec = update.get("spec")
            data = spec.model_dump(mode="json") if spec is not None else {"error": update.get("error")}
            pending.append("state-after-pm.json")
            yield TeamEvent(kind="spec", data=data)
        elif node == "consolidate_node":
            data = update["consolidation"].model_dump(mode="json")
            self._write_json("consolidation.json", data)
            yield TeamEvent(kind="consolidation", data=data)
        elif node == "dispatch_node":
            h = update["handoff"]
            assert self.run_dir is not None
            (self.run_dir / f"handoff-{h.task_id}.json").write_text(h.model_dump_json(indent=2))
            yield TeamEvent(kind="handoff", task_id=h.task_id, data=h.model_dump(mode="json"))
        elif node == "coder_node" and update.get("coder_step") == "close":
            ((task_id, record),) = update["task_status"].items()
            pending.append(f"state-after-{task_id}.json")
            yield TeamEvent(kind="task_done", task_id=task_id, data=record.model_dump(mode="json"))
        elif node == "finalize" and update.get("regression") is not None:
            self._write_json("regression.json", update["regression"])

    def _append_a2a(self, record: dict) -> None:
        assert self.run_dir is not None
        with (self.run_dir / A2A_LOG).open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _write_json(self, name: str, data: dict) -> None:
        assert self.run_dir is not None
        (self.run_dir / name).write_text(json.dumps(data, indent=2))

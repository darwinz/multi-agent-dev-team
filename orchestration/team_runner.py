"""TeamRunner: runs the team graph under one workspace lock and writes evidence files (spec §11)."""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import get_args
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


class TeamRunner:
    def __init__(self, settings: Settings, *, pm: PMAgent | None = None, coder: CoderAgent | None = None,
                 qa: QAAgent | None = None, broker_factory: Callable[[], Broker] | None = None) -> None:
        self.settings = settings
        self.broker_factory = broker_factory or Broker  # one fresh broker per run
        self.pm = pm or PMAgent(settings)
        self.coder = coder or CoderAgent(settings)
        self.qa = qa or QAAgent(settings)
        self.run_dir: Path | None = None
        self.final_state: dict | None = None
        self.cost_report_path: Path | None = None

    def reset_memory(self) -> None:
        self.coder.reset_memory()  # takes the RunLock itself: never call while stream() is running

    def run(self, requirement: str) -> dict:
        for _ in self.stream(requirement):
            pass
        assert self.final_state is not None
        return self.final_state

    def stream(self, requirement: str, *, api_check: ApiCheck | None = None,
               demo_check: DemoCheck | None = None) -> Iterator[TeamEvent]:
        s = self.settings
        self.run_dir = None
        self.final_state = None
        self.cost_report_path = None
        # Lock first: a busy workspace must leave no run directory (and no cost report) behind
        with RunLock(s.state_dir):
            run_id = uuid4().hex[:12]
            run_dir = s.state_dir / "runs" / run_id
            run_dir.mkdir(parents=True)
            self.run_dir = run_dir
            ctx = RunContext(run_id=run_id)
            started = _now()
            values: dict = {}
            outcome: str | None = None
            error: dict | None = None
            trace = None  # the root run tree while tracing; its URL is resolved once, at run end
            trace_resolved = False
            # run_log_file inside run_scope, so events.jsonl keeps only this run's records
            with run_scope(ctx), run_log_file(run_dir / "events.jsonl"):
                try:
                    log_event("run_start", requirement_chars=len(requirement),
                              models={r: s.model_for(r) for r in ROLES})
                    with root_trace(run_id, requirement) as trace:
                        (run_dir / "requirement.txt").write_text(requirement)
                        # The capability advertisements open the A2A log (spec §5)
                        self._append_a2a(_capabilities("coder", CODER_ACCEPTS))
                        self._append_a2a(_capabilities("qa", QA_ACCEPTS))
                        yield TeamEvent(kind="start", data={"run_id": run_id, "run_dir": str(run_dir)})

                        deps = TeamDeps(pm=self.pm, coder=self.coder, qa=self.qa, broker=self.broker_factory(),
                                        qa_max_iterations=s.qa_max_iterations)
                        graph = build_team_graph(deps)
                        pending: list[str] = []  # snapshot files waiting for the step's post-update values
                        for chunk in graph.stream(
                            initial_project_state(requirement),
                            stream_mode=["updates", "custom", "values"],
                            version="v2",
                            config={"recursion_limit": team_recursion_limit(s.qa_max_iterations)},
                        ):
                            kind, data = chunk["type"], chunk["data"]
                            if kind == "values":
                                values = data
                                for name in pending:
                                    self._write_json(name, state_to_jsonable(values))
                                pending.clear()
                            elif kind == "custom":
                                event = TeamEvent.model_validate(data)
                                self._on_custom(event)
                                yield event
                            elif kind == "updates":
                                for node, update in data.items():
                                    for event in self._on_update(node, update or {}, pending):
                                        yield event

                        state = state_to_jsonable(values)
                        for name in pending:  # defensive: the graph always ends with a values chunk
                            self._write_json(name, state)
                        # Inside the root, after the graph: the root run is posted by now, and the wait is bounded
                        trace_resolved = True
                        self._resolve_trace(ctx, trace)

                        # Still under the lock, so the checks and the evidence describe the same workspace
                        api: dict | None = None
                        demo: dict | None = None
                        if state["status"] != "failed":
                            if api_check is not None:
                                passed, report = api_check(s.workspace_dir, sorted(state["code_output"]))
                                (run_dir / "api-check.txt").write_text(report)
                                api = {"passed": passed, "report": report}
                            if demo_check is not None:
                                passed, report = demo_check(s.workspace_dir, state)
                                (run_dir / "demo-check.txt").write_text(report)
                                demo = {"passed": passed, "report": report}
                        outcome = OUTCOME[state["status"]]
                        self._write_json("final-state.json", state)
                        self.final_state = state
                        # outcome is already set: a consumer that closes the generator after `final` is no abort
                        yield TeamEvent(kind="final", data={"state": state, "api_check": api, "demo_check": demo})
                except BaseException as e:  # incl. GeneratorExit (closed early) and KeyboardInterrupt
                    if outcome is None:
                        outcome = "aborted"
                        error = {"type": type(e).__name__, "message": redact(str(e))}
                        log_event("run_aborted", level=logging.ERROR, **error)
                        self._write_json("final-state.json",
                                         state_to_jsonable(values or initial_project_state(requirement)))
                    raise
                finally:
                    if trace is not None and not trace_resolved:  # aborted: still try, bounded, never raising
                        with contextlib.suppress(Exception):
                            self._resolve_trace(ctx, trace)
                    report = build_cost_report(
                        run_id=run_id, started_at=started, finished_at=_now(), requirement=requirement,
                        outcome=outcome or "aborted", models={r: s.model_for(r) for r in ROLES},
                        calls=list(ctx.calls), trace_url=ctx.trace_url, error=error)
                    written, errors = write_cost_report(report, s.cost_reports_dir, run_dir)
                    for e in errors:  # logged, never raised: the run's own outcome or exception proceeds
                        log_event("cost_report_failed", level=logging.ERROR, type=type(e).__name__,
                                  message=redact(str(e)))
                    docs_copy, run_copy = s.cost_reports_dir / f"{run_id}.json", run_dir / "cost-report.json"
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

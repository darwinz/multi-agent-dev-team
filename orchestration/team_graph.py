"""The PM → Coder ↔ QA team graph (spec §4).

pm_node → [consolidate_node] → dispatch_node → coder_node (deliver) → reflect_node → qa_node → coder_node
(revise → qa_node, or close → dispatch_node / finalize). All Coder ↔ QA traffic goes over A2A.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.coder_agent import CoderAgent, ReflectionError
from agents.events import TeamEvent
from agents.pm_agent import PMAgent, PMError
from agents.qa_agent import QAAgent
from agents.result import CoderResult
from orchestration.a2a import (
    CODER_ACCEPTS,
    QA_ACCEPTS,
    Approved,
    Broker,
    Endpoint,
    FinalReport,
    FixInstruction,
    ReviewRequest,
)
from orchestration.handoff import make_handoff, qa_test_file, render_fix, render_revision, render_task
from orchestration.project_state import MAX_TASKS, ProjectState, TaskArtifact, TaskRecord
from orchestration.review import Issue, QAReport, Reflection
from tools.workspace import Workspace, sha256_file

_SUMMARY_CHARS = 500
NOOP_RETRY = "Your previous revision changed no files; make the changes now."
_OUTCOME = {"max_iterations": "unresolved", "qa_error": "qa_error", "no_revision": "no_revision"}


@dataclass
class TeamDeps:
    pm: PMAgent
    coder: CoderAgent  # one instance for the whole project: memory and workspace carry across tasks
    qa: QAAgent
    broker: Broker = field(default_factory=Broker)
    qa_max_iterations: int = 5


def team_recursion_limit(qa_max_iterations: int) -> int:
    """Per task: dispatch + deliver + reflect, then a qa_node and a coder_node per QA round (spec §9)."""
    return MAX_TASKS * (2 * qa_max_iterations + 3) + 10


def route_after_pm(state: ProjectState) -> Literal["consolidate_node", "dispatch_node", "finalize"]:
    if state["error"] or not state["tasks"]:
        return "finalize"
    return "consolidate_node" if len(state["tasks"]) > MAX_TASKS else "dispatch_node"


def route_after_coder(state: ProjectState) -> Literal["reflect_node", "qa_node", "dispatch_node", "finalize"]:
    if state["coder_step"] == "deliver":
        return "reflect_node"
    if state["coder_step"] == "revise":
        return "qa_node"
    return "dispatch_node" if state["current_task_index"] < len(state["tasks"]) else "finalize"


def _request_files(
    task_files: dict[str, str], workspace: Workspace
) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """Re-read every path from disk: (task_files without vanished paths, {path: sha256}, dropped paths)."""
    kept: dict[str, str] = {}
    shas: dict[str, str] = {}
    dropped: list[str] = []
    for path in sorted(task_files):
        p = workspace.root / path
        if p.is_file():
            kept[path] = p.read_bytes().decode("utf-8", errors="replace")
            shas[path] = sha256_file(p)
        else:
            dropped.append(path)
    return kept, shas, dropped


def _issue_line(issue: Issue) -> str:
    return f"{issue.test or issue.location}: {issue.assertion}"


def _after_runs(record: TaskRecord, results: list[CoderResult], task_files: dict[str, str]) -> TaskRecord:
    """The task's record after some Coder runs: turns add up, the latest run describes the task."""
    update: dict = {"iterations": record.iterations + sum(r.iterations for r in results),
                    "files": sorted(task_files)}
    if results:
        last = results[-1]
        update.update(coder_status=last.status, run_id=last.run_id, summary=last.explanation[:_SUMMARY_CHARS])
    return record.model_copy(update=update)


def _run_updates(task_id: str, results: list[CoderResult]) -> dict:
    """The artifacts and code_output updates for some Coder runs."""
    code: dict[str, str] = {}
    for r in results:
        code.update(r.files)
    return {"artifacts": [TaskArtifact(task_id=task_id, result=r) for r in results], "code_output": code}


def _dropped_note(task_id: str, dropped: list[str]) -> list[str]:
    return [f"{task_id}: dropped from review (no longer on disk): {', '.join(dropped)}"] if dropped else []


def _qa_crash(task_id: str, rnd: int, error: Exception) -> QAReport:
    """The report for a QA review that raised: no tests counted, never approved."""
    return QAReport(
        task_id=task_id, round=rnd, test_file=qa_test_file(task_id), suite_sha="", suites_run={}, tested_files={},
        tests_run=0, passed=0, failed=0, errors=0, skipped=0, missing=[], exit_code=None, timed_out=False,
        duration_s=0.0, results=[], issues=[], tampered=False,
        qa_error=f"QA crashed: {type(error).__name__}: {error}", approved=False,
    )


def build_team_graph(deps: TeamDeps) -> CompiledStateGraph:
    broker = deps.broker
    coder_ep = Endpoint("coder", CODER_ACCEPTS, broker)
    qa_ep = Endpoint("qa", QA_ACCEPTS, broker)
    workspace = deps.coder.workspace

    def forward_a2a(record: dict) -> None:
        task_id = record["message"]["payload"].get("task_id")
        get_stream_writer()(TeamEvent(kind="a2a", task_id=task_id, data=record).model_dump())

    broker.on_event = forward_a2a  # every send, receive and rejection becomes an `a2a` stream event

    def _emit(kind: str, task_id: str, data: dict) -> None:
        get_stream_writer()(TeamEvent(kind=kind, task_id=task_id, data=data).model_dump())

    def _run_coder(task_id: str, text: str) -> CoderResult:
        result: CoderResult | None = None
        for event in deps.coder.stream(text, lock_held=True):
            _emit("coder", task_id, event.model_dump())
            if event.kind == "result":
                result = CoderResult.model_validate(event.data["result"])
        assert result is not None, "Coder stream ended without a result event"
        return result

    def pm_node(state: ProjectState) -> dict:
        try:
            spec = deps.pm.draft_spec(state["requirement"])
        except PMError as e:
            return {"error": str(e), "status": "failed", "work_in_progress": [f"PM failed: {e}"]}
        # task_status merges (keys can't be removed), so seed only the ids that will execute;
        # when the spec is consolidated, consolidate_node seeds the merged ids instead
        seed = ({t.task_id: TaskRecord(task_id=t.task_id, status="pending") for t in spec.tasks}
                if len(spec.tasks) <= MAX_TASKS else {})
        return {
            "spec": spec,
            "tasks": spec.tasks,
            "task_status": seed,
            "work_in_progress": [f"PM drafted spec '{spec.name}' with {len(spec.tasks)} task(s)"],
        }

    def consolidate_node(state: ProjectState) -> dict:
        # the merged ids (M1..Mk) are what executes, so they are the only pending records seeded
        tasks, consolidation = deps.pm.consolidate(state["spec"])
        return {
            "tasks": tasks,
            "consolidation": consolidation,
            "task_status": {t.task_id: TaskRecord(task_id=t.task_id, status="pending") for t in tasks},
            "work_in_progress": [f"Consolidated {len(consolidation.original_tasks)} tasks into {len(tasks)} "
                                 f"({consolidation.method})"],
        }

    def dispatch_node(state: ProjectState) -> dict:
        h = make_handoff(state)
        return {
            "handoff": h,
            "task_status": {h.task_id: TaskRecord(task_id=h.task_id, status="in_progress")},
            "qa_round": 0, "outstanding_request": None, "task_files": {}, "qa_suite": None,
            "coder_step": None, "reflection_ok": False,
            "work_in_progress": [f"Handed {h.task_id} ({h.position} of {h.total}) to the Coder: {h.title}"],
        }

    def coder_node(state: ProjectState) -> dict:
        h = state["handoff"]  # the only task input: the Coder never sees spec, tasks or requirement
        record = state["task_status"][h.task_id]
        if state["qa_round"] == 0:
            result = _run_coder(h.task_id, render_task(h))
            task_files = {**state["task_files"], **result.files}
            return {
                **_run_updates(h.task_id, [result]),
                "task_files": task_files,
                "task_status": {h.task_id: _after_runs(record, [result], task_files)},
                "coder_step": "deliver",
                "work_in_progress": [f"{h.task_id} delivered (Coder: {result.status}, "
                                     f"{result.iterations} iteration(s))"],
            }

        msg = coder_ep.receive(task_id=h.task_id, correlation_id=state["outstanding_request"])
        if msg.intent == "fix_instruction":
            fix = FixInstruction(**msg.payload)
            result = _run_coder(h.task_id, render_fix(h, fix))
            task_files, shas, dropped = _request_files({**state["task_files"], **result.files}, workspace)
            rnd = state["qa_round"] + 1
            req = coder_ep.send("qa", "review_request", ReviewRequest(task_id=h.task_id, round=rnd, files=shas))
            return {
                **_run_updates(h.task_id, [result]),
                "task_files": task_files,
                "task_status": {h.task_id: _after_runs(record, [result], task_files)},
                "outstanding_request": req.correlation_id,
                "coder_step": "revise",
                "work_in_progress": [f"{h.task_id} revised after QA round {fix.round} (Coder: {result.status}); "
                                     f"review_request round {rnd}", *_dropped_note(h.task_id, dropped)],
            }

        # approved or final_report: the only intents that close a task (course rule)
        update: dict = {}
        if msg.intent == "approved" and not state["reflection_ok"]:
            # QA never approves without a recorded revision; this guard keeps `done` honest regardless
            status, outcome = "failed", "no_revision"
            unresolved = ["QA approved, but self-reflection recorded no revision, so the task cannot be done"]
        elif msg.intent == "approved":
            status, outcome, unresolved = "done", "approved", []
            update["approved_suites"] = {h.task_id: state["qa_suite"]}
        else:
            report = FinalReport(**msg.payload)
            status, outcome = "failed", _OUTCOME[report.reason]
            unresolved = [_issue_line(i) for i in report.unresolved]
        closed = record.model_copy(update={"status": status, "qa_outcome": outcome, "unresolved": unresolved})
        return {
            **update,
            "task_status": {h.task_id: closed},
            "current_task_index": state["current_task_index"] + 1,
            "handoff": None,
            "outstanding_request": None,
            "coder_step": "close",
            "work_in_progress": [f"{h.task_id} {status} (QA: {outcome} after {record.qa_rounds} round(s))"],
        }

    def reflect_node(state: ProjectState) -> dict:
        """Critique, then a revision run; a revision must change a file (spec §8). One retry."""
        h = state["handoff"]
        task_files, before, dropped = _request_files(state["task_files"], workspace)
        after, changed = before, []
        critique, results, errors = None, [], []
        ok, noop = False, False
        for _ in range(2):
            try:
                critique = deps.coder.critique(render_task(h), task_files)
            except ReflectionError as e:
                errors.append(str(e))
                noop = False
                continue
            text = render_revision(h, critique) + (f"\n\n{NOOP_RETRY}" if noop else "")
            result = _run_coder(h.task_id, text)
            results.append(result)
            task_files, after, gone = _request_files({**task_files, **result.files}, workspace)
            dropped += gone
            changed = sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p))
            if changed:
                ok = True
                break
            errors.append("the revision changed no files")
            noop = True

        reflection = Reflection(
            task_id=h.task_id, critique=critique, before=before, after=after, changed=changed,
            revision_status=results[-1].status if results else None,
            error=None if ok else "; ".join(errors),
        )
        _emit("reflection", h.task_id, {**reflection.model_dump(mode="json"), "dropped": dropped})
        req = coder_ep.send("qa", "review_request", ReviewRequest(task_id=h.task_id, round=1, files=after))
        record = _after_runs(state["task_status"][h.task_id], results, task_files).model_copy(
            update={"reflected": ok})
        verdict = f"revised {', '.join(changed)}" if ok else f"no revision ({reflection.error})"
        return {
            **_run_updates(h.task_id, results),
            "task_files": task_files,
            "task_status": {h.task_id: record},
            "reflections": [reflection],
            "reflection_ok": ok,
            "outstanding_request": req.correlation_id,
            "work_in_progress": [f"{h.task_id} self-reflection: {verdict}; review_request round 1",
                                 *_dropped_note(h.task_id, dropped)],
        }

    def qa_node(state: ProjectState) -> dict:
        h = state["handoff"]
        req = qa_ep.receive(task_id=h.task_id)
        rnd = state["qa_round"] + 1
        try:
            report, suite = deps.qa.review(h, ReviewRequest(**req.payload), round=rnd, suite=state["qa_suite"],
                                           prior=state["approved_suites"],
                                           later=state["tasks"][state["current_task_index"] + 1:])
        except Exception as e:  # any QA crash ends the task as qa_error instead of crashing the run
            report, suite = _qa_crash(h.task_id, rnd, e), state["qa_suite"]
        _emit("qa_report", h.task_id, report.model_dump(mode="json"))
        intent, payload = _reply(state, report, rnd)
        qa_ep.send("coder", intent, payload, correlation_id=req.correlation_id)
        record = state["task_status"][h.task_id].model_copy(
            update={"qa_rounds": rnd, "qa_test_file": report.test_file})
        return {
            "qa_round": rnd,
            "qa_suite": suite,
            "qa_reports": [report],
            "task_status": {h.task_id: record},
            "work_in_progress": [f"{h.task_id} QA round {rnd}: {report.passed}/{report.tests_run} passed, "
                                 f"{len(report.issues)} issue(s) → {intent}"],
        }

    def _reply(state: ProjectState, report: QAReport, rnd: int) -> tuple[str, Approved | FinalReport | FixInstruction]:
        task_id = report.task_id
        score = f"round {rnd}: {report.passed}/{report.tests_run} QA tests passed, {len(report.issues)} issue(s)"

        def final(reason: str, why: str) -> tuple[str, FinalReport]:
            return "final_report", FinalReport(task_id=task_id, round=rnd, reason=reason, unresolved=report.issues,
                                               summary=f"{why}; {score}")

        if not state["reflection_ok"]:
            return final("no_revision", "self-reflection recorded no revision, so the task cannot be approved")
        if report.approved:
            return "approved", Approved(task_id=task_id, round=rnd, tests_run=report.tests_run,
                                        test_file=report.test_file)
        if report.qa_error:
            return final("qa_error", f"QA failed: {report.qa_error}")
        if rnd >= deps.qa_max_iterations:
            return final("max_iterations", f"reached the limit of {deps.qa_max_iterations} QA round(s)")
        issues = report.issues or [Issue(  # an unapproved report always carries a High issue; never send none
            severity="High", test=None, criterion=None, location=report.test_file,
            assertion="QA did not approve this round", fix="Make every QA test pass")]
        return "fix_instruction", FixInstruction(task_id=task_id, round=rnd, issues=issues, summary=score)

    def finalize(state: ProjectState) -> dict:
        if state["error"]:
            return {"status": "failed", "work_in_progress": ["Run failed"]}
        records = dict(state["task_status"])
        done = [t.task_id for t in state["tasks"] if records[t.task_id].status == "done"]
        regression: dict | None = None
        notes: list[str] = []
        regressed: dict[str, TaskRecord] = {}
        if done:
            # Project regression check (spec §4): every done task's canonical suite, against the final workspace
            report = deps.qa.run_suites({i: state["approved_suites"][i] for i in done}, task_id="regression", round=0)
            regression = report.model_dump(mode="json")
            unowned = [i for i in report.issues if not i.assertion.startswith("regression in ")]
            for task_id in done:
                module, prefix = qa_test_file(task_id)[:-3] + "::", f"regression in {task_id}: "
                issues = [i for i in report.issues if i.assertion.startswith(prefix)] + unowned
                bad = [r for r in report.results if r.name.startswith(module) and r.outcome != "passed"]
                if issues or bad:
                    lines = [_issue_line(i) for i in issues] or [f"{r.name}: {r.outcome}" for r in bad]
                    regressed[task_id] = records[task_id].model_copy(
                        update={"status": "failed", "qa_outcome": "regressed", "unresolved": lines})
            notes.append(f"Regression check over {', '.join(done)}: {report.passed}/{report.tests_run} passed"
                         + (f"; regressed: {', '.join(regressed)}" if regressed else ""))
        records.update(regressed)
        status = "completed" if all(records[t.task_id].status == "done" for t in state["tasks"]) else "partial"
        return {"status": status, "task_status": regressed, "regression": regression,
                "work_in_progress": [*notes, f"Run {status}"]}

    g = StateGraph(ProjectState)
    g.add_node("pm_node", pm_node)
    g.add_node("consolidate_node", consolidate_node)
    g.add_node("dispatch_node", dispatch_node)
    g.add_node("coder_node", coder_node)
    g.add_node("reflect_node", reflect_node)
    g.add_node("qa_node", qa_node)
    g.add_node("finalize", finalize)
    g.add_edge(START, "pm_node")
    g.add_conditional_edges("pm_node", route_after_pm, ["consolidate_node", "dispatch_node", "finalize"])
    g.add_edge("consolidate_node", "dispatch_node")
    g.add_edge("dispatch_node", "coder_node")
    g.add_conditional_edges("coder_node", route_after_coder, ["reflect_node", "qa_node", "dispatch_node", "finalize"])
    g.add_edge("reflect_node", "qa_node")
    g.add_edge("qa_node", "coder_node")
    g.add_edge("finalize", END)
    return g.compile()

"""The PM → Coder team graph: pm_node → [consolidate_node] → dispatch_node ⇄ coder_node → finalize."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.coder_agent import CoderAgent
from agents.events import TeamEvent
from agents.pm_agent import PMAgent, PMError
from agents.result import CoderResult
from orchestration.handoff import make_handoff, render_task
from orchestration.project_state import MAX_TASKS, ProjectState, TaskArtifact, TaskRecord

_SUMMARY_CHARS = 500


@dataclass
class TeamDeps:
    pm: PMAgent
    coder: CoderAgent  # one instance for the whole project: memory and workspace carry across tasks


def route_after_pm(state: ProjectState) -> Literal["consolidate_node", "dispatch_node", "finalize"]:
    if state["error"] or not state["tasks"]:
        return "finalize"
    return "consolidate_node" if len(state["tasks"]) > MAX_TASKS else "dispatch_node"


def route_after_coder(state: ProjectState) -> Literal["dispatch_node", "finalize"]:
    return "dispatch_node" if state["current_task_index"] < len(state["tasks"]) else "finalize"


def build_team_graph(deps: TeamDeps) -> CompiledStateGraph:
    def pm_node(state: ProjectState) -> dict:
        try:
            spec = deps.pm.draft_spec(state["requirement"])
        except PMError as e:
            return {"error": str(e), "status": "failed", "work_in_progress": [f"PM failed: {e}"]}
        return {
            "spec": spec,
            "tasks": spec.tasks,
            "task_status": {t.task_id: TaskRecord(task_id=t.task_id, status="pending") for t in spec.tasks},
            "work_in_progress": [f"PM drafted spec '{spec.name}' with {len(spec.tasks)} task(s)"],
        }

    def consolidate_node(state: ProjectState) -> dict:
        # task_status merges, so the original ids' pending records remain; only ids in `tasks` are judged
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
            "work_in_progress": [f"Handed {h.task_id} ({h.position} of {h.total}) to the Coder: {h.title}"],
        }

    def coder_node(state: ProjectState) -> dict:
        h = state["handoff"]  # the only input: the Coder never sees spec, tasks or requirement
        writer = get_stream_writer()
        result: CoderResult | None = None
        for event in deps.coder.stream(render_task(h), lock_held=True):
            writer(TeamEvent(kind="coder", task_id=h.task_id, data=event.model_dump()).model_dump())
            if event.kind == "result":
                result = CoderResult.model_validate(event.data["result"])
        assert result is not None, "Coder stream ended without a result event"
        status = "done" if result.status == "success" else "failed"
        record = TaskRecord(
            task_id=h.task_id, status=status, coder_status=result.status, run_id=result.run_id,
            iterations=result.iterations, files=sorted(result.files),
            summary=result.explanation[:_SUMMARY_CHARS],
        )
        return {
            "task_status": {h.task_id: record},
            "artifacts": [TaskArtifact(task_id=h.task_id, result=result)],
            "code_output": result.files,
            "current_task_index": state["current_task_index"] + 1,
            "handoff": None,
            "work_in_progress": [f"{h.task_id} {status} (Coder: {result.status}, {result.iterations} iteration(s))"],
        }

    def finalize(state: ProjectState) -> dict:
        if state["error"]:
            status = "failed"
        elif all(state["task_status"][t.task_id].status == "done" for t in state["tasks"]):
            status = "completed"
        else:
            status = "partial"
        return {"status": status, "work_in_progress": [f"Run {status}"]}

    g = StateGraph(ProjectState)
    g.add_node("pm_node", pm_node)
    g.add_node("consolidate_node", consolidate_node)
    g.add_node("dispatch_node", dispatch_node)
    g.add_node("coder_node", coder_node)
    g.add_node("finalize", finalize)
    g.add_edge(START, "pm_node")
    g.add_conditional_edges("pm_node", route_after_pm, ["consolidate_node", "dispatch_node", "finalize"])
    g.add_edge("consolidate_node", "dispatch_node")
    g.add_edge("dispatch_node", "coder_node")
    g.add_conditional_edges("coder_node", route_after_coder, ["dispatch_node", "finalize"])
    g.add_edge("finalize", END)
    return g.compile()

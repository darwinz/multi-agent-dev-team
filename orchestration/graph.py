"""The hand-built bounded ReAct graph: retrieve_memory → agent → tools → … → finalize."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.events import EventKind, StepEvent
from agents.prompts import SYSTEM_PROMPT
from memory.context import format_memory_context
from memory.semantic import SemanticMemory
from memory.window import SlidingWindow
from orchestration.finalize import build_result
from orchestration.routing import Decision, classify, extract_plan
from orchestration.state import CoderState, ExecInfo, ToolEvent, WriteInfo
from tools.dispatch import ToolContext, run_action
from tools.workspace import ToolError

NUDGE = (
    "Observation: no tool was called. Call a tool to make progress, or call finish (alone) "
    "when the deliverable has been executed successfully."
)
SKIPPED = "skipped: an earlier call in this batch failed"
_QUERY_CHARS = 500


@dataclass
class GraphDeps:
    model: BaseChatModel  # already bound to TOOL_SPECS
    tool_ctx: ToolContext
    semantic: SemanticMemory
    window: SlidingWindow
    max_iterations: int
    top_k: int
    max_distance: float
    run_id: str


def initial_state(task: str) -> CoderState:
    return CoderState(
        task=task, messages=[HumanMessage(task)], iterations=0, seq=0, plan=[], events=[], writes={},
        executions=[], memory_context="", memories=[], decision=None, finish=None, stop_reason=None,
        result=None,
    )


def build_graph(deps: GraphDeps) -> CompiledStateGraph:
    def emit(kind: EventKind, iteration: int, **data: Any) -> None:
        get_stream_writer()(StepEvent(kind=kind, iteration=iteration, max_iterations=deps.max_iterations,
                                      data=data).model_dump())

    def retrieve_memory(state: CoderState) -> dict:
        query = state["task"]
        if len(state["messages"]) > 1:
            last = state["messages"][-1]
            if isinstance(last, (ToolMessage, HumanMessage)):
                query = f"{query}\n{last.text}"
        hits = deps.semantic.search(query[:_QUERY_CHARS], k=deps.top_k, max_distance=deps.max_distance)
        turns = deps.window.turns()
        hit_dicts = [h.model_dump() for h in hits]
        emit("memory", state["iterations"] + 1, hits=hit_dicts, window_turns=len(turns))
        return {"memory_context": format_memory_context(turns, hits), "memories": hit_dicts}

    def agent(state: CoderState) -> dict:
        n = state["iterations"] + 1
        system = SystemMessage(
            SYSTEM_PROMPT.format(max_iterations=deps.max_iterations) + "\n\n" + state["memory_context"]
        )
        message: AIMessage = deps.model.invoke([system, *state["messages"]])
        update: dict = {"messages": [message], "iterations": n}
        if message.text:
            emit("thought", n, text=message.text)
        plan = extract_plan(message.text)
        if plan:
            update["plan"] = plan
            emit("plan", n, steps=plan)

        decision = classify(message, n, deps.max_iterations)
        update["decision"] = asdict(decision)
        if decision.kind == "finish":
            update["finish"] = decision.finish_args
            update["stop_reason"] = "finished"
        elif decision.kind == "limit":
            update["stop_reason"] = "iteration_limit"
            emit("limit", n, unexecuted=[c["name"] for c in message.tool_calls])
        return update

    def route(state: CoderState) -> str:
        return "finalize" if state["decision"]["kind"] in ("finish", "limit") else "tools"

    def tools(state: CoderState) -> dict:
        n = state["iterations"]
        d = Decision(**state["decision"])
        messages: list = [ToolMessage(obs, tool_call_id=call_id) for call_id, obs in d.rejections]
        events: list[ToolEvent] = []
        writes: dict[str, WriteInfo] = {}
        executions: list[ExecInfo] = []
        seq = state["seq"]
        failed = False
        for call in d.run_calls:
            if failed:
                messages.append(ToolMessage(SKIPPED, tool_call_id=call["id"]))
                continue
            seq += 1
            emit("action", n, tool=call["name"], args=call["args"])
            execution: ExecInfo | None = None
            try:
                out = run_action(deps.tool_ctx, call["name"], call["args"])
                obs, ok = out.observation, True
                if out.write:
                    writes[out.write.path] = WriteInfo(seq=seq, sha256=out.write.sha256)
                if out.execution:
                    e = out.execution
                    execution = ExecInfo(
                        seq=seq, target=e.target, ok=e.ok, exit_code=e.exit_code, timed_out=e.timed_out,
                        stdout=e.stdout, stderr=e.stderr, duration_s=e.duration_s, loaded=e.loaded,
                    )
                    executions.append(execution)
            except ToolError as err:
                obs, ok, failed = f"error: {err}", False, True
            events.append(ToolEvent(seq=seq, iteration=n, name=call["name"], args=call["args"], observation=obs, ok=ok))
            messages.append(ToolMessage(obs, tool_call_id=call["id"], status="success" if ok else "error"))
            emit("observation", n, tool=call["name"], ok=ok, text=obs, execution=execution)
        if d.kind == "nudge":
            messages.append(HumanMessage(NUDGE))
            emit("nudge", n, text=NUDGE)
        return {"messages": messages, "events": events, "writes": writes, "executions": executions, "seq": seq}

    def finalize(state: CoderState) -> dict:
        result = build_result(state, run_id=deps.run_id, workspace=deps.tool_ctx.workspace,
                              max_iterations=deps.max_iterations)
        dumped = result.model_dump()
        emit("result", state["iterations"], result=dumped)
        return {"result": dumped}

    g = StateGraph(CoderState)
    g.add_node("retrieve_memory", retrieve_memory)
    g.add_node("agent", agent)
    g.add_node("tools", tools)
    g.add_node("finalize", finalize)
    g.add_edge(START, "retrieve_memory")
    g.add_edge("retrieve_memory", "agent")
    g.add_conditional_edges("agent", route, ["tools", "finalize"])
    g.add_edge("tools", "retrieve_memory")
    g.add_edge("finalize", END)
    return g.compile()


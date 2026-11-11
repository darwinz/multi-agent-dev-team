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
FINISH_NUDGE = (
    "Observation: finish was rejected (see above). Fix its arguments and call it alone, or call a tool "
    "to make progress."
)
SKIPPED = "skipped: an earlier call in this batch failed"
_QUERY_CHARS = 500


@dataclass
class GraphDeps:
    """What the Coder graph's nodes share: the tool-bound model, tools, both memory layers and the limits."""

    model: BaseChatModel  # already bound to TOOL_SPECS
    tool_ctx: ToolContext
    semantic: SemanticMemory
    window: SlidingWindow
    max_iterations: int
    top_k: int
    max_distance: float
    run_id: str


def initial_state(task: str) -> CoderState:
    """The Coder graph's starting state for `task`: the task as the first message, nothing run yet."""
    return CoderState(
        task=task, messages=[HumanMessage(task)], iterations=0, seq=0, plan=[], events=[], writes={},
        executions=[], memory_context="", memories=[], decision=None, finish=None, stop_reason=None,
        result=None,
    )


def _to_exec_info(seq: int, e: Any) -> ExecInfo:
    """The state record of one `exec_python` run, numbered by its tool-call sequence."""
    return ExecInfo(seq=seq, target=e.target, ok=e.ok, exit_code=e.exit_code, timed_out=e.timed_out,
                    stdout=e.stdout, stderr=e.stderr, duration_s=e.duration_s, loaded=e.loaded)


class _CoderNodes:
    """The Coder graph's node functions, sharing the model, tools and memory in `GraphDeps`."""

    def __init__(self, deps: GraphDeps) -> None:
        self.deps = deps

    def emit(self, kind: EventKind, iteration: int, **data: Any) -> None:
        """Send one `StepEvent` to the LangGraph custom stream."""
        get_stream_writer()(StepEvent(kind=kind, iteration=iteration, max_iterations=self.deps.max_iterations,
                                      data=data).model_dump())

    def retrieve_memory(self, state: CoderState) -> dict:
        """Search semantic memory with the task (plus the latest observation) before every model call."""
        query = state["task"]
        if len(state["messages"]) > 1:
            last = state["messages"][-1]
            if isinstance(last, (ToolMessage, HumanMessage)):
                query = f"{query}\n{last.text}"
        hits = self.deps.semantic.search(query[:_QUERY_CHARS], k=self.deps.top_k, max_distance=self.deps.max_distance)
        turns = self.deps.window.turns()
        hit_dicts = [h.model_dump() for h in hits]
        self.emit("memory", state["iterations"] + 1, hits=hit_dicts, window_turns=len(turns))
        return {"memory_context": format_memory_context(turns, hits), "memories": hit_dicts}

    def agent(self, state: CoderState) -> dict:
        """One model call; `classify` decides whether to run tools, nudge, finish or stop at the limit."""
        n = state["iterations"] + 1
        system = SystemMessage(
            SYSTEM_PROMPT.format(max_iterations=self.deps.max_iterations) + "\n\n" + state["memory_context"]
        )
        message: AIMessage = self.deps.model.invoke([system, *state["messages"]])
        update: dict = {"messages": [message], "iterations": n}
        if message.text:
            self.emit("thought", n, text=message.text)
        plan = extract_plan(message.text)
        if plan:
            update["plan"] = plan
            self.emit("plan", n, steps=plan)

        decision = classify(message, n, self.deps.max_iterations)
        update["decision"] = asdict(decision)
        if decision.kind == "finish":
            update["finish"] = decision.finish_args
            update["stop_reason"] = "finished"
        elif decision.kind == "limit":
            update["stop_reason"] = "iteration_limit"
            rejected_ids = {call_id for call_id, _ in decision.rejections}
            self.emit("limit", n,
                      unexecuted=[c["name"] for c in message.tool_calls if c["id"] not in rejected_ids],
                      rejected=[obs.splitlines()[0] for _, obs in decision.rejections])
        return update

    @staticmethod
    def route(state: CoderState) -> str:
        """After `agent`: finalize on finish or the iteration limit, otherwise run the tools."""
        return "finalize" if state["decision"]["kind"] in ("finish", "limit") else "tools"

    def _run_call(self, call: dict, seq: int, n: int, out_state: dict) -> bool:
        """Run one tool call, recording its event, writes and execution in `out_state`; returns whether it ran ok."""
        self.emit("action", n, tool=call["name"], args=call["args"])
        execution: ExecInfo | None = None
        try:
            out = run_action(self.deps.tool_ctx, call["name"], call["args"])
            obs, ok = out.observation, True
            if out.write:
                out_state["writes"][out.write.path] = WriteInfo(seq=seq, sha256=out.write.sha256)
            for w in out.side_writes:  # written by the code that just ran: needs a later run to verify
                out_state["writes"][w.path] = WriteInfo(seq=seq, sha256=w.sha256)
            if out.execution:
                execution = _to_exec_info(seq, out.execution)
                out_state["executions"].append(execution)
        except ToolError as err:
            obs, ok = f"error: {err}", False
        out_state["events"].append(
            ToolEvent(seq=seq, iteration=n, name=call["name"], args=call["args"], observation=obs, ok=ok))
        out_state["messages"].append(ToolMessage(obs, tool_call_id=call["id"], status="success" if ok else "error"))
        self.emit("observation", n, tool=call["name"], ok=ok, text=obs, execution=execution)
        return ok

    def tools(self, state: CoderState) -> dict:
        """Run the decided tool calls in order; after the first failure the rest of the batch is skipped."""
        n = state["iterations"]
        d = Decision(**state["decision"])
        out: dict = {
            "messages": [ToolMessage(obs, tool_call_id=call_id, status="error") for call_id, obs in d.rejections],
            "events": [], "writes": {}, "executions": [],
        }
        seq = state["seq"]
        failed = False
        for call in d.run_calls:
            if failed:
                out["messages"].append(ToolMessage(SKIPPED, tool_call_id=call["id"], status="error"))
                continue
            seq += 1
            failed = not self._run_call(call, seq, n, out)
        if d.kind == "nudge":
            nudge = FINISH_NUDGE if d.rejections else NUDGE
            out["messages"].append(HumanMessage(nudge))
            self.emit("nudge", n, text=nudge)
        return {**out, "seq": seq}

    def finalize(self, state: CoderState) -> dict:
        """Build the validated `CoderResult` from the run's state and emit it."""
        result = build_result(state, run_id=self.deps.run_id, workspace=self.deps.tool_ctx.workspace,
                              max_iterations=self.deps.max_iterations)
        dumped = result.model_dump()
        self.emit("result", state["iterations"], result=dumped)
        return {"result": dumped}


def build_graph(deps: GraphDeps) -> CompiledStateGraph:
    """Compile the bounded ReAct graph: retrieve_memory → agent → (tools → retrieve_memory)* → finalize."""
    nodes = _CoderNodes(deps)
    g = StateGraph(CoderState)
    g.add_node("retrieve_memory", nodes.retrieve_memory)
    g.add_node("agent", nodes.agent)
    g.add_node("tools", nodes.tools)
    g.add_node("finalize", nodes.finalize)
    g.add_edge(START, "retrieve_memory")
    g.add_edge("retrieve_memory", "agent")
    g.add_conditional_edges("agent", nodes.route, ["tools", "finalize"])
    g.add_edge("tools", "retrieve_memory")
    g.add_edge("finalize", END)
    return g.compile()

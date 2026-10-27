"""CoderAgent: one bounded ReAct run per task, with a workspace lock and two memory layers."""

from __future__ import annotations

import ast
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

from langchain_core.language_models import BaseChatModel

from agents.config import Settings
from agents.events import StepEvent
from agents.model import make_model
from agents.result import CoderResult, ExecutionResult
from memory.semantic import SemanticMemory
from memory.window import SlidingWindow
from orchestration.graph import GraphDeps, build_graph, initial_state
from tools.dispatch import ToolContext
from tools.specs import TOOL_SPECS
from tools.workspace import ResetGeneration, RunLock, Workspace


def _defines(files: dict[str, str]) -> list[str]:
    names = []
    for path, src in files.items():
        if not path.endswith(".py"):
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        names += [
            f"{path}:{node.name}"
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
    return names


def memory_document(task: str, result: CoderResult) -> str:
    """The text embedded into semantic memory after each run."""
    return "\n".join([
        f"Task: {task}",
        f"Outcome: {result.status}",
        f"Files: {', '.join(result.files) or 'none'}",
        f"Plan: {' | '.join(result.plan)}",
        f"Summary: {result.explanation[:500]}",
        f"Defines: {', '.join(_defines(result.files)) or 'none'}",
    ])


class CoderAgent:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        model: BaseChatModel | None = None,
        embedding_function: Any = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.model = model
        self.workspace = Workspace(self.settings.workspace_dir)
        self.window = SlidingWindow(self.settings.window_turns)
        self.semantic = SemanticMemory(self.settings.chroma_dir, embedding_function=embedding_function)
        self._generation = ResetGeneration(self.settings.state_dir)
        self.generation_seen = self._generation.read()
        self.last_written: set[str] = set()

    def _bound_model(self) -> Any:
        if self.model is None:
            self.model = make_model(self.settings)
        return self.model.bind_tools(TOOL_SPECS)

    def stream(self, task: str) -> Iterator[StepEvent]:
        s = self.settings
        with RunLock(s.state_dir):
            current = self._generation.read()
            if current != self.generation_seen:
                self.window.clear()
                self.semantic.reopen()  # another process recreated the collection
                self.generation_seen = current

            run_id = uuid.uuid4().hex[:12]
            yield StepEvent(kind="start", iteration=0, max_iterations=s.max_iterations,
                            data={"task": task, "run_id": run_id, "model": s.model})
            self.last_written = set()
            state: dict = {}
            try:
                deps = GraphDeps(
                    model=self._bound_model(),
                    tool_ctx=ToolContext(self.workspace, s.exec_timeout_s, s.state_dir / "exec"),
                    semantic=self.semantic,
                    window=self.window,
                    max_iterations=s.max_iterations,
                    top_k=s.memory_top_k,
                    max_distance=s.memory_max_distance,
                    run_id=run_id,
                )
                for chunk in build_graph(deps).stream(
                    initial_state(task),
                    config={"recursion_limit": 3 * s.max_iterations + 5},
                    stream_mode=["custom", "values"],
                    version="v2",
                ):
                    if chunk["type"] == "values":
                        state = chunk["data"]
                        continue
                    event = StepEvent.model_validate(chunk["data"])
                    if event.kind == "action" and event.data.get("tool") == "write_file":
                        try:
                            self.last_written.add(self.workspace.relpath(self.workspace.resolve(event.data["args"]["path"])))
                        except Exception:  # the observation reports the bad path; nothing to highlight
                            pass
                    yield event
            except Exception as e:  # model/provider/graph failure: report it as an error result
                yield StepEvent(kind="error", iteration=state.get("iterations", 0), max_iterations=s.max_iterations,
                                data={"message": f"{type(e).__name__}: {e}"})
                result = CoderResult(
                    code="",
                    explanation=f"Run failed: {type(e).__name__}: {e}",
                    plan=state.get("plan") or ["Run failed before a plan was recorded."],
                    result=ExecutionResult(executed=False, target=None, exit_code=None),
                    status="error",
                    iterations=state.get("iterations", 0),
                    files={},
                    memories_used=[],
                    run_id=run_id,
                    verified=False,
                    error=str(e),
                )
                yield StepEvent(kind="result", iteration=result.iterations, max_iterations=s.max_iterations,
                                data={"result": result.model_dump()})
                return

            result = CoderResult.model_validate(state["result"])
            self.window.add(task, result.explanation)
            try:
                self.semantic.add(memory_document(task, result), {
                    "run_id": run_id,
                    "status": result.status,
                    "files": ",".join(result.files),
                    "created_at": datetime.now(UTC).isoformat(),
                })
            except Exception as e:  # e.g. the embedding model can't download: keep the result
                yield StepEvent(kind="error", iteration=result.iterations, max_iterations=s.max_iterations,
                                data={"message": f"memory write-back failed: {type(e).__name__}: {e}"})

    def run(self, task: str) -> CoderResult:
        result = None
        for event in self.stream(task):
            if event.kind == "result":
                result = CoderResult.model_validate(event.data["result"])
        assert result is not None, "stream ended without a result event"
        return result

    def reset_memory(self) -> None:
        """Clear both memory layers and the workspace; other processes drop their buffers on next run."""
        with RunLock(self.settings.state_dir):
            self.semantic.clear()
            self.workspace.clear()
            self.generation_seen = self._generation.bump()
            self.window.clear()


def run_coder(task: str, settings: Settings | None = None) -> CoderResult:
    return CoderAgent(settings).run(task)

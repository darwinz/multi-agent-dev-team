"""CoderAgent: one bounded ReAct run per task, with a workspace lock and two memory layers."""

from __future__ import annotations

import ast
import contextlib
import shutil
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agents.config import Settings
from agents.events import StepEvent
from agents.model import make_model, resilient
from agents.prompts import CRITIQUE_PROMPT
from agents.result import CoderResult, ExecutionResult
from memory.semantic import SemanticMemory
from memory.window import SlidingWindow
from orchestration.finalize import code_block
from orchestration.graph import GraphDeps, build_graph, initial_state
from orchestration.review import Critique
from tools.dispatch import ToolContext
from tools.specs import TOOL_SPECS
from tools.workspace import ResetGeneration, RunLock, Workspace


class ReflectionError(Exception):
    """The Coder could not produce a valid self-critique."""


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
    """The Coder: a bounded ReAct agent with two memory layers, run on one shared workspace."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        model: BaseChatModel | None = None,
        embedding_function: Any = None,
    ) -> None:
        """Set up memory (window + Chroma), the workspace and the reset counter; the model is built on first use."""
        self.settings = settings or Settings.from_env()
        self.model = resilient(model, self.settings, "coder") if model is not None else None
        self.workspace = Workspace(self.settings.workspace_dir)
        self.window = SlidingWindow(self.settings.window_turns)
        self.semantic = SemanticMemory(self.settings.chroma_dir, embedding_function=embedding_function,
                                       host=self.settings.chroma_host, port=self.settings.chroma_port)
        self._generation = ResetGeneration(self.settings.state_dir)
        self.generation_seen = self._generation.read()
        self.last_written: set[str] = set()

    def _bound_model(self) -> Any:
        if self.model is None:
            self.model = make_model(self.settings, "coder")
        return self.model.bind_tools(TOOL_SPECS)  # ReAct steps are never response-cached

    def critique(self, task_text: str, files: dict[str, str]) -> Critique:
        """One structured call on the unbound model (no tools); one retry with the parse error appended."""
        try:
            if self.model is None:
                self.model = make_model(self.settings, "coder")
            structured = self.model.with_structured_output(Critique, include_raw=True)
        except Exception as e:  # missing provider package, bad model id
            raise ReflectionError(f"critique model setup failed: {type(e).__name__}: {e}") from e
        code = "\n\n".join(f"--- {path} ---\n{text}" for path, text in sorted(files.items())) or "(no files)"
        messages: list[Any] = [SystemMessage(CRITIQUE_PROMPT),
                               HumanMessage(f"{task_text}\n\nYour files:\n{code}")]
        error = ""
        for attempt in range(2):
            try:
                out = structured.invoke(messages)
            except Exception as e:  # provider failures end the attempt, no retry
                raise ReflectionError(f"critique model call failed: {type(e).__name__}: {e}") from e
            if out.get("parsing_error") is None and out.get("parsed") is not None:
                return out["parsed"]
            error = str(out.get("parsing_error") or "the model returned no structured critique")
            if attempt == 0:
                messages.append(HumanMessage(
                    f"Your critique was invalid:\n{error}\n\nReturn a corrected critique that fixes these problems."))
        raise ReflectionError(f"critique invalid twice: {error}")

    def stream(self, task: str, *, lock_held: bool = False) -> Iterator[StepEvent]:
        """Run one task, yielding its `StepEvent`s and ending with a `result` event; then write memory back."""
        s = self.settings
        with contextlib.nullcontext() if lock_held else RunLock(s.state_dir):
            self._sync_generation()
            run_id = uuid.uuid4().hex[:12]
            yield StepEvent(kind="start", iteration=0, max_iterations=s.max_iterations,
                            data={"task": task, "run_id": run_id, "model": s.model})
            self.last_written = set()
            state: dict = {}
            try:
                yield from self._stream_graph(task, run_id, state)
            except Exception as e:  # noqa: BLE001 - model/provider/graph failure: report it as an error result
                yield StepEvent(kind="error", iteration=state.get("iterations", 0), max_iterations=s.max_iterations,
                                data={"message": f"{type(e).__name__}: {e}"})
                result = self._error_result(e, state, run_id)
                yield StepEvent(kind="result", iteration=result.iterations, max_iterations=s.max_iterations,
                                data={"result": result.model_dump()})
                return
            result = CoderResult.model_validate(state["result"])
            yield from self._remember(task, result, run_id)

    def _sync_generation(self) -> None:
        """Drop the window and reopen Chroma if another process reset memory since this agent last looked."""
        current = self._generation.read()
        if current != self.generation_seen:
            self.window.clear()
            self.semantic.reopen()  # another process recreated the collection
            self.generation_seen = current

    def _stream_graph(self, task: str, run_id: str, state: dict) -> Iterator[StepEvent]:
        """Stream the ReAct graph's events; `state` is kept equal to the graph's latest full state."""
        s = self.settings
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
                state.clear()
                state.update(chunk["data"])
                continue
            event = StepEvent.model_validate(chunk["data"])
            if event.kind == "action" and event.data.get("tool") == "write_file":
                self._note_write(event.data["args"].get("path"))
            yield event

    def _note_write(self, path: object) -> None:
        """Remember a write_file target so a failed run can still report what it wrote."""
        # the observation reports a bad path; there is nothing to highlight
        with contextlib.suppress(Exception):
            self.last_written.add(self.workspace.relpath(self.workspace.resolve(path)))

    def _error_result(self, e: Exception, state: dict, run_id: str) -> CoderResult:
        """The `error` result for a run the model, provider or graph broke: never verified."""
        files = self._written_files()
        return CoderResult(
            code=code_block(files) if files else "",
            explanation=f"Run failed: {type(e).__name__}: {e}",
            plan=state.get("plan") or ["Run failed before a plan was recorded."],
            result=ExecutionResult(executed=False, target=None, exit_code=None),
            status="error",
            iterations=state.get("iterations", 0),
            files=files,
            memories_used=[],
            run_id=run_id,
            verified=False,
            error=str(e),
        )

    def _remember(self, task: str, result: CoderResult, run_id: str) -> Iterator[StepEvent]:
        """Add the run to both memory layers; a semantic-memory failure is reported, never raised."""
        self.window.add(task, result.explanation)
        try:
            self.semantic.add(memory_document(task, result), {
                "run_id": run_id,
                "status": result.status,
                "files": ",".join(result.files),
                "created_at": datetime.now(UTC).isoformat(),
            })
        except Exception as e:  # noqa: BLE001 - e.g. the embedding model can't download: keep the result
            yield StepEvent(kind="error", iteration=result.iterations, max_iterations=self.settings.max_iterations,
                            data={"message": f"memory write-back failed: {type(e).__name__}: {e}"})

    def _written_files(self) -> dict[str, str]:
        files = {}
        for path in sorted(self.last_written):
            p = self.workspace.root / path
            if p.is_file():
                files[path] = p.read_bytes().decode("utf-8", errors="replace")
        return files

    def run(self, task: str, *, lock_held: bool = False) -> CoderResult:
        """Run one task to completion and return its `CoderResult` (the non-streaming form of `stream`)."""
        result = None
        for event in self.stream(task, lock_held=lock_held):
            if event.kind == "result":
                result = CoderResult.model_validate(event.data["result"])
        assert result is not None, "stream ended without a result event"
        return result

    def reset_memory(self) -> None:
        """Clear both memory layers and the workspace; other processes drop their buffers on next run."""
        with RunLock(self.settings.state_dir):
            self.semantic.clear()
            self.workspace.clear()
            shutil.rmtree(self.settings.state_dir / "exec", ignore_errors=True)
            self.generation_seen = self._generation.bump()
            self.window.clear()


def run_coder(task: str, settings: Settings | None = None) -> CoderResult:
    """Convenience entry point: a fresh `CoderAgent` runs one task."""
    return CoderAgent(settings).run(task)

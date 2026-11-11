"""Build the validated CoderResult from state and disk. Makes no model call."""

from __future__ import annotations

from agents.result import CoderResult, ExecutionResult
from memory.semantic import MemoryHit
from orchestration.state import CoderState
from orchestration.verify import verify_delivery
from tools.workspace import Workspace


def code_block(files: dict[str, str]) -> str:
    """The delivered code: one file as-is, several files each under a `# ===== path =====` header."""
    if len(files) == 1:
        return next(iter(files.values()))
    return "\n\n".join(f"# ===== {path} =====\n{content}" for path, content in files.items())


def _files_on_disk(state: CoderState, workspace: Workspace) -> dict[str, str]:
    """The current content of every file the run wrote that still exists (read back from disk, not from state)."""
    files = {}
    for path in sorted(state["writes"]):
        p = workspace.root / path
        if p.is_file():
            files[path] = p.read_bytes().decode("utf-8", errors="replace")
    return files


def _last_execution(executions: list) -> tuple[ExecutionResult, bool]:
    """The run's last execution as an `ExecutionResult`, and whether it succeeded."""
    if not executions:
        return ExecutionResult(executed=False, target=None, exit_code=None), False
    last = executions[-1]
    return ExecutionResult(
        executed=True, target=last["target"], exit_code=last["exit_code"], stdout=last["stdout"],
        stderr=last["stderr"], timed_out=last["timed_out"], duration_s=last["duration_s"],
    ), last["ok"]


def _limit_explanation(state: CoderState, max_iterations: int) -> str:
    """The explanation for a run stopped at the iteration limit: its last action and observation."""
    last_event = state["events"][-1] if state["events"] else None
    return (
        f"Stopped after {state['iterations']} of {max_iterations} iterations without calling finish. "
        f"Last action: {last_event['name'] if last_event else 'none'}. "
        f"Last observation: {last_event['observation'][:300] if last_event else 'none'}."
    )


def build_result(state: CoderState, *, run_id: str, workspace: Workspace, max_iterations: int) -> CoderResult:
    """The run's validated `CoderResult`: `success` only if the last execution passed and delivery is verified."""
    files = _files_on_disk(state, workspace)
    executions = state["executions"]
    result, ran_ok = _last_execution(executions)
    finish = state.get("finish") or {}
    v = verify_delivery(state["writes"], executions, workspace, finish.get("entry_point"))
    notes = list(v.notes)

    if state.get("stop_reason") == "iteration_limit":
        status = "iteration_limit"
        explanation = _limit_explanation(state, max_iterations)
        plan = state["plan"] or ["No plan was recorded before the run stopped."]
    else:
        explanation, plan = finish["explanation"], finish["plan"]
        if ran_ok and v.verified:
            status = "success"
        else:
            status = "failed"
            if not executions:
                notes.append("code was never executed")
            elif not ran_ok:
                notes.append("last execution failed")

    return CoderResult(
        code=code_block(files),
        explanation=explanation,
        plan=plan,
        result=result,
        status=status,
        iterations=state["iterations"],
        files=files,
        memories_used=[MemoryHit.model_validate(h) for h in state["memories"]],
        run_id=run_id,
        verified=v.verified,
        verification_notes=notes,
    )

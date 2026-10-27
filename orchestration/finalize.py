"""Build the validated CoderResult from state and disk. Makes no model call."""

from __future__ import annotations

from agents.result import CoderResult, ExecutionResult
from memory.semantic import MemoryHit
from orchestration.state import CoderState
from orchestration.verify import verify_delivery
from tools.workspace import Workspace


def code_block(files: dict[str, str]) -> str:
    if len(files) == 1:
        return next(iter(files.values()))
    return "\n\n".join(f"# ===== {path} =====\n{content}" for path, content in files.items())


def build_result(state: CoderState, *, run_id: str, workspace: Workspace, max_iterations: int) -> CoderResult:
    files = {}
    for path in sorted(state["writes"]):
        p = workspace.root / path
        if p.is_file():
            files[path] = p.read_bytes().decode("utf-8", errors="replace")

    executions = state["executions"]
    if executions:
        last = executions[-1]
        result = ExecutionResult(
            executed=True, target=last["target"], exit_code=last["exit_code"], stdout=last["stdout"],
            stderr=last["stderr"], timed_out=last["timed_out"], duration_s=last["duration_s"],
        )
        ran_ok = last["ok"]
    else:
        result = ExecutionResult(executed=False, target=None, exit_code=None)
        ran_ok = False

    finish = state.get("finish") or {}
    v = verify_delivery(state["writes"], executions, workspace, finish.get("entry_point"))
    notes = list(v.notes)

    n = state["iterations"]
    if state.get("stop_reason") == "iteration_limit":
        status = "iteration_limit"
        last_event = state["events"][-1] if state["events"] else None
        explanation = (
            f"Stopped after {n} of {max_iterations} iterations without calling finish. "
            f"Last action: {last_event['name'] if last_event else 'none'}. "
            f"Last observation: {last_event['observation'][:300] if last_event else 'none'}."
        )
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
        iterations=n,
        files=files,
        memories_used=[MemoryHit.model_validate(h) for h in state["memories"]],
        run_id=run_id,
        verified=v.verified,
        verification_notes=notes,
    )

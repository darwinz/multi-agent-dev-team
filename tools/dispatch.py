"""Validate a tool call's arguments and run the matching action tool."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from tools.exec_python import ExecOutcome, exec_python
from tools.file_tools import WriteOutcome, read_file, write_file
from tools.specs import ACTION_TOOLS, ARG_MODELS
from tools.workspace import ToolError, Workspace, sha256_file


@dataclass
class ToolContext:
    workspace: Workspace
    exec_timeout_s: float
    record_dir: Path


@dataclass
class ActionOutcome:
    observation: str
    write: WriteOutcome | None = None
    execution: ExecOutcome | None = None
    side_writes: list[WriteOutcome] = field(default_factory=list)  # .py files the executed code changed


def run_action(ctx: ToolContext, name: str, args: dict) -> ActionOutcome:
    if name not in ACTION_TOOLS:
        raise ToolError(f"unknown tool: {name}")
    try:
        a = ARG_MODELS[name].model_validate(args)
    except ValidationError as e:
        raise ToolError(f"invalid arguments for {name}: {e}") from None

    try:
        return _dispatch(ctx, name, a)
    except ToolError:
        raise
    except (OSError, ValueError) as e:  # e.g. a directory in the way, an unencodable character
        raise ToolError(f"{name} failed: {type(e).__name__}: {e}") from None


def _dispatch(ctx: ToolContext, name: str, a) -> ActionOutcome:
    if name == "write_file":
        w = write_file(ctx.workspace, a.path, a.content)
        return ActionOutcome(f"wrote {w.path} ({w.lines} lines, {w.bytes} bytes)", write=w)
    if name == "read_file":
        return ActionOutcome(read_file(ctx.workspace, a.path))
    before = _py_snapshot(ctx.workspace)
    e = exec_python(
        ctx.workspace,
        record_dir=ctx.record_dir,
        path=a.path,
        code=a.code,
        # The configured timeout is a ceiling; the model may only ask for less
        timeout=min(a.timeout, ctx.exec_timeout_s) if a.timeout is not None else ctx.exec_timeout_s,
        mode=a.mode,
    )
    after = _py_snapshot(ctx.workspace)
    changed = [p for p, h in after.items() if before.get(p) != h]
    side_writes = [_outcome(ctx.workspace, p, after[p]) for p in changed]
    observation = e.to_observation()
    if changed:
        observation += f"\nfiles changed by this run: {', '.join(changed)}"
    return ActionOutcome(observation, execution=e, side_writes=side_writes)


def _py_snapshot(ws: Workspace) -> dict[str, str]:
    return {p: sha256_file(ws.root / p) for p in ws.list_files() if p.endswith(".py")}


def _outcome(ws: Workspace, path: str, sha: str) -> WriteOutcome:
    data = (ws.root / path).read_bytes()
    return WriteOutcome(path=path, sha256=sha, bytes=len(data), lines=len(data.splitlines()))

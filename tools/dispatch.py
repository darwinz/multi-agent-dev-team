"""Validate a tool call's arguments and run the matching action tool."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from tools.exec_python import ExecOutcome, exec_python
from tools.file_tools import WriteOutcome, read_file, write_file
from tools.specs import ACTION_TOOLS, ARG_MODELS
from tools.workspace import ToolError, Workspace


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


def run_action(ctx: ToolContext, name: str, args: dict) -> ActionOutcome:
    if name not in ACTION_TOOLS:
        raise ToolError(f"unknown tool: {name}")
    try:
        a = ARG_MODELS[name].model_validate(args)
    except ValidationError as e:
        raise ToolError(f"invalid arguments for {name}: {e}") from None

    if name == "write_file":
        w = write_file(ctx.workspace, a.path, a.content)
        return ActionOutcome(f"wrote {w.path} ({w.lines} lines, {w.bytes} bytes)", write=w)
    if name == "read_file":
        return ActionOutcome(read_file(ctx.workspace, a.path))
    e = exec_python(
        ctx.workspace,
        record_dir=ctx.record_dir,
        path=a.path,
        code=a.code,
        timeout=a.timeout if a.timeout is not None else ctx.exec_timeout_s,
        mode=a.mode,
    )
    return ActionOutcome(e.to_observation(), execution=e)

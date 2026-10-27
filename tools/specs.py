"""Tool argument models and the tool specs bound to the model."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

FINISH = "finish"
ACTION_TOOLS = ("read_file", "write_file", "exec_python")
MAX_CALL_TIMEOUT_S = 60.0


class ReadFileArgs(BaseModel):
    path: str = Field(description="File path, relative to the workspace directory.")


class WriteFileArgs(BaseModel):
    path: str = Field(description="File path, relative to the workspace directory.")
    content: str = Field(description="Full file content (UTF-8). Overwrites any existing file.")


class ExecPythonArgs(BaseModel):
    path: str | None = Field(None, description="Python file to run, relative to the workspace directory.")
    code: str | None = Field(None, description="Inline Python to run instead of a file.")
    timeout: float | None = Field(
        None, gt=0, le=MAX_CALL_TIMEOUT_S,
        description="Seconds before the process tree is killed. Defaults to the configured timeout.",
    )
    mode: Literal["auto", "script", "pytest"] = Field(
        "auto", description="auto runs test_*.py / *_test.py under pytest and anything else as a script.",
    )


class FinishArgs(BaseModel):
    explanation: str = Field(min_length=1, description="What you built, how it works, and how you verified it.")
    plan: list[str] = Field(min_length=1, description="The plan you followed, one step per item.")
    entry_point: str | None = Field(
        None, description="The script or test file that demonstrates the deliverable, relative to the workspace.",
    )


_DESCRIPTIONS = {
    "read_file": (ReadFileArgs, "Read a text file from the workspace. Paths are relative to the workspace."),
    "write_file": (WriteFileArgs, "Create or overwrite a file in the workspace. Paths are relative to the workspace."),
    "exec_python": (
        ExecPythonArgs,
        "Run a Python file (or inline code) from the workspace in a subprocess, with the workspace as the "
        "working directory. Paths are relative to the workspace. Test files run under pytest. Returns exit "
        "code, stdout, stderr and whether the timeout killed it.",
    ),
    FINISH: (
        FinishArgs,
        "Finish the task. Call it alone (no other tool calls in the same response), after your last action "
        "and after the delivered code has run successfully. Paths are relative to the workspace.",
    ),
}

ARG_MODELS: dict[str, type[BaseModel]] = {name: model for name, (model, _) in _DESCRIPTIONS.items()}

TOOL_SPECS: list[dict] = [
    {"name": name, "description": desc, "parameters": model.model_json_schema()}
    for name, (model, desc) in _DESCRIPTIONS.items()
]

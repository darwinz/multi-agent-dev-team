"""The Coder agent's validated structured output."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agents.config import MAX_ITERATIONS_CAP
from memory.semantic import MemoryHit

Status = Literal["success", "failed", "iteration_limit", "error"]


class ExecutionResult(BaseModel):
    """The last real exec_python run of this task (or `executed=False` if nothing ran)."""

    executed: bool
    target: str | None
    exit_code: int | None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    duration_s: float = 0.0


class CoderResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(description="Contents of the delivered files, read back from disk.")
    explanation: str = Field(min_length=1)
    plan: list[str] = Field(min_length=1)
    result: ExecutionResult
    status: Status
    iterations: int = Field(ge=0, le=MAX_ITERATIONS_CAP)
    files: dict[str, str]
    memories_used: list[MemoryHit]
    run_id: str
    verified: bool
    verification_notes: list[str] = []
    error: str | None = None

    @model_validator(mode="after")
    def _success_is_earned(self) -> CoderResult:
        r = self.result
        if self.status == "success" and not (
            r.executed and r.exit_code == 0 and not r.timed_out and self.verified
        ):
            raise ValueError("success requires a successful execution and verified delivery")
        return self

"""Runtime configuration: environment variables → validated Settings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

MAX_ITERATIONS_CAP = 10

# Settings field → environment variable
_ENV_VARS = {
    "project_dir": "CODER_PROJECT_DIR",
    "model": "CODER_MODEL",
    "max_iterations": "CODER_MAX_ITERATIONS",
    "exec_timeout_s": "EXEC_TIMEOUT_SECONDS",
    "window_turns": "MEMORY_WINDOW_TURNS",
    "memory_top_k": "MEMORY_TOP_K",
    "memory_max_distance": "MEMORY_MAX_DISTANCE",
    "max_tokens": "CODER_MAX_TOKENS",
}


class Settings(BaseModel):
    project_dir: Path
    model: str = "anthropic:claude-sonnet-5-5"
    max_iterations: int = MAX_ITERATIONS_CAP
    exec_timeout_s: float = Field(10.0, gt=0)
    window_turns: int = Field(10, ge=1)
    memory_top_k: int = Field(3, ge=1)
    memory_max_distance: float = Field(0.65, gt=0)
    max_tokens: int = Field(8192, ge=1)

    @field_validator("max_iterations")
    @classmethod
    def _cap_iterations(cls, v: int) -> int:
        if not 1 <= v <= MAX_ITERATIONS_CAP:
            raise ValueError(f"max_iterations must be between 1 and {MAX_ITERATIONS_CAP}, got {v}")
        return v

    @property
    def workspace_dir(self) -> Path:
        return self.project_dir / "workspace"

    @property
    def chroma_dir(self) -> Path:
        return self.project_dir / ".chroma"

    @property
    def state_dir(self) -> Path:
        return self.project_dir / ".coder"

    @classmethod
    def from_env(cls, **overrides: Any) -> Settings:
        """Build settings from the environment (and `.env`); non-None overrides win."""
        load_dotenv()
        values: dict[str, Any] = {}
        for field, var in _ENV_VARS.items():
            raw = os.environ.get(var)
            if raw:
                values[field] = raw
        values.setdefault("project_dir", Path.cwd())
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)

"""Runtime configuration: environment variables → validated Settings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

MAX_ITERATIONS_CAP = 10
QA_MAX_ITERATIONS_CAP = 5  # the course caps the Coder <-> QA loop at five test runs

Role = Literal["pm", "coder", "qa"]

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
    "qa_max_iterations": "QA_MAX_ITERATIONS",
    "pm_model": "PM_MODEL",
    "qa_model": "QA_MODEL",
    "retry_max": "RETRY_MAX",
    "retry_base_delay_s": "RETRY_BASE_DELAY_S",
    "retry_max_delay_s": "RETRY_MAX_DELAY_S",
    "breaker_failure_threshold": "BREAKER_FAILURE_THRESHOLD",
    "breaker_cooldown_s": "BREAKER_COOLDOWN_S",
    "cache_ttl_s": "CACHE_TTL_S",
    "chroma_host": "CHROMA_HOST",
    "chroma_port": "CHROMA_PORT",
    "log_level": "LOG_LEVEL",
}


class Settings(BaseModel):
    project_dir: Path
    model: str = "anthropic:claude-haiku-5-5"  # the Coder role (CODER_MODEL)
    pm_model: str = "anthropic:claude-sonnet-5-5"
    qa_model: str = "anthropic:claude-haiku-5-5"
    max_iterations: int = MAX_ITERATIONS_CAP
    exec_timeout_s: float = Field(10.0, gt=0)
    window_turns: int = Field(10, ge=1)
    memory_top_k: int = Field(3, ge=1)
    memory_max_distance: float = Field(0.65, gt=0)
    max_tokens: int = Field(8192, ge=1)
    qa_max_iterations: int = Field(5, ge=1, le=QA_MAX_ITERATIONS_CAP)
    retry_max: int = Field(3, ge=0, le=3)
    retry_base_delay_s: float = Field(1.0, ge=0)
    retry_max_delay_s: float = Field(30.0, ge=0)
    breaker_failure_threshold: int = Field(3, ge=1)
    breaker_cooldown_s: float = Field(30.0, ge=0)
    cache_ttl_s: float = Field(3600.0, ge=0)
    chroma_host: str | None = None
    chroma_port: int = 8000
    log_level: str = "INFO"

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

    @property
    def cache_dir(self) -> Path:
        return self.state_dir / "cache"

    @property
    def cost_reports_dir(self) -> Path:
        return self.project_dir / "docs" / "cost_reports"

    def model_for(self, role: Role) -> str:
        return {"pm": self.pm_model, "coder": self.model, "qa": self.qa_model}[role]

    @classmethod
    def from_env(cls, **overrides: Any) -> Settings:
        """Build settings from the environment (and the project's `.env`); non-None overrides win."""
        project_dir = overrides.get("project_dir") or os.environ.get("CODER_PROJECT_DIR") or Path.cwd()
        # Only the project's own .env: load_dotenv()'s default upward search would pick up an ancestor's keys
        load_dotenv(Path(project_dir) / ".env")
        values: dict[str, Any] = {"project_dir": project_dir}
        for field, var in _ENV_VARS.items():
            raw = os.environ.get(var)
            if raw:
                values[field] = raw
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)

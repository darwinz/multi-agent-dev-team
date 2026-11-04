"""Shared state and data models for the PM -> Coder project graph."""

from __future__ import annotations

import operator
import re
from collections.abc import Mapping
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agents.result import CoderResult, Status
from orchestration.review import QAReport, QATestSuite, Reflection

Priority = Literal["high", "medium", "low"]
TaskStatus = Literal["pending", "in_progress", "done", "failed"]
RunStatus = Literal["running", "completed", "partial", "failed"]
QAOutcome = Literal["approved", "unresolved", "qa_error", "no_revision", "regressed"]
CoderStep = Literal["deliver", "revise", "close"]

MAX_TASKS = 8


class AcceptanceCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str  # "AC1", "AC2", ... unique within a spec
    text: str  # testable statement of required behaviour


# Task ids name evidence files (handoff-<id>.json), so the PM may only use T1, T2, ...
TASK_ID_RE = re.compile(r"T[1-9][0-9]*")


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str  # "T1", "T2", ... unique within the list
    title: str
    description: str
    acceptance_criteria: list[str] = Field(min_length=1)  # task-level, each checkable by running code
    priority: Priority
    covers: list[str] = Field(min_length=1)  # ids of spec-level criteria this task satisfies


class Spec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    acceptance_criteria: list[AcceptanceCriterion] = Field(min_length=1)
    tasks: list[Task] = Field(min_length=1)  # the PM's original decomposition

    @model_validator(mode="after")
    def _consistent(self) -> Spec:
        problems: list[str] = []
        crit_ids = self.criterion_ids()
        task_ids = [t.task_id for t in self.tasks]
        for label, ids in (("criterion", crit_ids), ("task", task_ids)):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            if dupes:
                problems.append(f"duplicate {label} ids: {', '.join(dupes)}")
        bad = [repr(i) for i in task_ids if not TASK_ID_RE.fullmatch(i)]
        if bad:
            problems.append(f"unsupported task ids {', '.join(bad)}: each task id must be T1, T2, ... (T and a number)")
        known = set(crit_ids)
        unknown = sorted({c for t in self.tasks for c in t.covers if c not in known})
        if unknown:
            problems.append(f"covers references unknown criterion ids: {', '.join(unknown)}")
        covered = {c for t in self.tasks for c in t.covers}
        uncovered = [i for i in crit_ids if i not in covered]
        if uncovered:
            problems.append(f"criteria not covered by any task: {', '.join(uncovered)}")
        if problems:
            raise ValueError("; ".join(problems))
        return self

    def criterion_ids(self) -> list[str]:
        return [c.id for c in self.acceptance_criteria]


class TaskHandoff(BaseModel):
    """The readiness signal handed from dispatch_node to coder_node."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    title: str
    description: str
    acceptance_criteria: list[str]
    priority: Priority
    position: int  # 1-based index in the execution list
    total: int
    spec_name: str
    spec_description: str
    prior_files: list[str]  # workspace files produced by earlier tasks


class TaskRecord(BaseModel):
    """Per-task progress."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    status: TaskStatus
    coder_status: Status | None = None  # the CoderResult status
    run_id: str | None = None
    iterations: int = 0  # Coder turns across all of the task's runs
    files: list[str] = []
    summary: str = ""  # the latest CoderResult.explanation, truncated
    qa_rounds: int = 0
    qa_outcome: QAOutcome | None = None
    unresolved: list[str] = []  # the issues still open when the task closed
    reflected: bool = False  # self-reflection recorded a revision
    qa_test_file: str | None = None


class Consolidation(BaseModel):
    """Requirement-mapping evidence for a merged task list."""

    model_config = ConfigDict(extra="forbid")

    original_tasks: list[Task]
    merged_from: dict[str, list[str]]  # new task_id -> original task_ids, a partition of the originals
    method: Literal["llm", "deterministic"]


class TaskArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    result: CoderResult


def merge_dicts(a: dict, b: dict) -> dict:
    """Reducer: later keys win."""
    return {**a, **b}


class ProjectState(TypedDict):
    requirement: str  # set at entry, never modified
    spec: Spec | None  # written once by pm_node
    tasks: list[Task]  # execution list (<= MAX_TASKS); = spec.tasks unless consolidated
    consolidation: Consolidation | None
    task_status: Annotated[dict[str, TaskRecord], merge_dicts]
    current_task_index: int
    handoff: TaskHandoff | None  # set by dispatch_node, cleared by coder_node when the task closes
    artifacts: Annotated[list[TaskArtifact], operator.add]  # one per Coder run
    code_output: Annotated[dict[str, str], merge_dicts]  # accumulated path -> content, later tasks win
    work_in_progress: Annotated[list[str], operator.add]  # chronological progress log
    error: str | None
    status: RunStatus
    # --- the per-task Coder <-> QA loop (reset by dispatch_node unless noted) ---
    qa_round: int  # QA rounds completed for the current task
    outstanding_request: str | None  # correlation id of the Coder's open review_request
    task_files: dict[str, str]  # every file this task's Coder runs wrote; merged by the nodes, no reducer
    qa_suite: QATestSuite | None
    approved_suites: Annotated[dict[str, QATestSuite], merge_dicts]  # never reset within a run
    regression: dict | None  # the finalize regression run (QAReport dump), None if no task was done
    reflection_ok: bool  # the current task's reflection recorded a revision
    coder_step: CoderStep | None
    qa_reports: Annotated[list[QAReport], operator.add]
    reflections: Annotated[list[Reflection], operator.add]


def initial_project_state(requirement: str) -> ProjectState:
    return ProjectState(
        requirement=requirement,
        spec=None,
        tasks=[],
        consolidation=None,
        task_status={},
        current_task_index=0,
        handoff=None,
        artifacts=[],
        code_output={},
        work_in_progress=[],
        error=None,
        status="running",
        qa_round=0,
        outstanding_request=None,
        task_files={},
        qa_suite=None,
        approved_suites={},
        regression=None,
        reflection_ok=False,
        coder_step=None,
        qa_reports=[],
        reflections=[],
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def state_to_jsonable(state: Mapping) -> dict:
    """Plain-JSON view of a project state (pydantic values dumped with mode="json")."""
    return {k: _jsonable(v) for k, v in state.items()}

"""Coverage-preserving consolidation: group the PM's tasks into at most MAX_TASKS merged tasks.

A model (or the deterministic fallback) only proposes the grouping, a title and a summary per group.
build_merged builds everything that carries requirements, so a merge cannot drop one.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from orchestration.project_state import Consolidation, Spec, Task

_RANK = {"high": 2, "medium": 1, "low": 0}


class MergeGroup(BaseModel):
    """One proposed merged task: a title, a one-line summary and the original task ids it absorbs."""

    model_config = ConfigDict(extra="forbid")

    title: str
    summary: str  # one line
    source_ids: list[str] = Field(min_length=1)  # original task ids, in original order


class MergePlan(BaseModel):
    """A proposed grouping of the original tasks into merged tasks (from the model or deterministic)."""

    model_config = ConfigDict(extra="forbid")

    groups: list[MergeGroup]


def validate_merge(spec: Spec, plan: MergePlan, max_tasks: int) -> list[str]:
    """Problems with a plan; empty means valid. Flattened source_ids must equal the original ids exactly."""
    problems: list[str] = []
    if len(plan.groups) > max_tasks:
        problems.append(f"{len(plan.groups)} groups exceeds the limit of {max_tasks}")
    expected = [t.task_id for t in spec.tasks]
    flat = [i for g in plan.groups for i in g.source_ids]
    if flat != expected:
        for pos in range(max(len(flat), len(expected))):
            got = flat[pos] if pos < len(flat) else None
            want = expected[pos] if pos < len(expected) else None
            if got != want:
                problems.append(
                    f"source_ids must list every original task once, contiguously and in order; "
                    f"at position {pos + 1} expected {want}, got {got}")
                break
    return problems


def build_merged(
    spec: Spec, groups: list[MergeGroup], method: Literal["llm", "deterministic"]
) -> tuple[list[Task], Consolidation]:
    """Merge each group into one Task, keeping every source criterion, description and cover."""
    by_id = {t.task_id: t for t in spec.tasks}
    crit_order = spec.criterion_ids()
    merged: list[Task] = []
    merged_from: dict[str, list[str]] = {}
    for n, group in enumerate(groups, 1):
        sources = [by_id[i] for i in group.source_ids]
        covered = {c for s in sources for c in s.covers}
        descriptions = "\n".join(f"[{s.task_id}] {s.description}" for s in sources)
        task_id = f"M{n}"  # distinct from the original T-ids, which spec.tasks keeps
        merged.append(Task(
            task_id=task_id,
            title=group.title,
            description=f"{group.summary}\n\n{descriptions}",
            acceptance_criteria=[c for s in sources for c in s.acceptance_criteria],
            priority=max((s.priority for s in sources), key=_RANK.__getitem__),
            covers=[c for c in crit_order if c in covered],
        ))
        merged_from[task_id] = list(group.source_ids)
    return merged, Consolidation(original_tasks=list(spec.tasks), merged_from=merged_from, method=method)


def deterministic_merge(spec: Spec, max_tasks: int) -> tuple[list[Task], Consolidation]:
    """Split the tasks evenly into at most max_tasks contiguous groups, earlier groups larger."""
    n = len(spec.tasks)
    k = min(max_tasks, n)
    base, extra = divmod(n, k)
    groups: list[MergeGroup] = []
    start = 0
    for g in range(k):
        size = base + (1 if g < extra else 0)
        chunk = spec.tasks[start:start + size]
        start += size
        groups.append(MergeGroup(title=" + ".join(t.title for t in chunk), summary="",
                                 source_ids=[t.task_id for t in chunk]))
    return build_merged(spec, groups, "deterministic")

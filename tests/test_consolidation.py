"""Consolidation: coverage-preserving merge to <= MAX_TASKS, LLM grouping with deterministic fallback."""

from __future__ import annotations

import pytest

from agents.pm_agent import PMAgent
from orchestration.consolidation import (
    MergeGroup,
    MergePlan,
    build_merged,
    deterministic_merge,
    validate_merge,
)
from orchestration.project_state import Spec
from tests.fakes import ScriptedChatModel, make_spec, structured

RANK = {"high": 2, "medium": 1, "low": 0}


def ids(n: int) -> list[str]:
    return [f"T{i}" for i in range(1, n + 1)]


def plan_of(*groups: list[str]) -> MergePlan:
    return MergePlan(groups=[MergeGroup(title=f"G{i}", summary=f"Group {i}", source_ids=g)
                             for i, g in enumerate(groups, 1)])


def good_plan() -> MergePlan:
    t = ids(11)
    return plan_of(t[0:2], t[2:4], t[4:6], t[6:7], t[7:8], t[8:9], t[9:10], t[10:11])


def mixed_spec() -> Spec:
    spec = make_spec(11, 4)
    prios = ["low", "medium", "high"] * 4
    tasks = [t.model_copy(update={"priority": prios[i]}) for i, t in enumerate(spec.tasks)]
    return Spec(**{**spec.model_dump(), "tasks": [t.model_dump() for t in tasks]})


def test_deterministic_merge_partitions_in_order():
    spec = mixed_spec()
    tasks, cons = deterministic_merge(spec, 8)
    assert [t.task_id for t in tasks] == [f"M{i}" for i in range(1, 9)]
    assert [len(cons.merged_from[t.task_id]) for t in tasks] == [2, 2, 2, 1, 1, 1, 1, 1]
    assert [o for t in tasks for o in cons.merged_from[t.task_id]] == ids(11)
    assert cons.method == "deterministic"
    by_id = {t.task_id: t for t in spec.tasks}
    crit_order = spec.criterion_ids()
    for t in tasks:
        srcs = [by_id[o] for o in cons.merged_from[t.task_id]]
        union = {c for s in srcs for c in s.covers}
        assert t.covers == [c for c in crit_order if c in union]
        assert t.priority == max((s.priority for s in srcs), key=RANK.__getitem__)


def test_build_merged_keeps_all_requirement_text():
    spec = make_spec(11, 4)
    tasks, cons = build_merged(spec, good_plan().groups, "llm")
    by_id = {t.task_id: t for t in spec.tasks}
    for merged in tasks:
        for oid in cons.merged_from[merged.task_id]:
            src = by_id[oid]
            for crit in src.acceptance_criteria:
                assert crit in merged.acceptance_criteria
            assert f"[{oid}] {src.description}" in merged.description


def test_merged_ids_do_not_collide_with_originals():
    spec = make_spec(11, 4)
    tasks, cons = build_merged(spec, good_plan().groups, "llm")
    assert [t.task_id for t in tasks] == [f"M{i}" for i in range(1, len(tasks) + 1)]
    assert set(cons.merged_from) == {t.task_id for t in tasks}


def test_llm_merge_cannot_drop_criteria(settings):
    spec = make_spec(11, 4)
    spec.tasks[4].acceptance_criteria.append("returns 422 on invalid input")
    model = ScriptedChatModel(responses=[structured("MergePlan", good_plan().model_dump())])
    tasks, cons = PMAgent(settings, model=model).consolidate(spec)
    assert cons.method == "llm"
    owner = next(t for t in tasks if "T5" in cons.merged_from[t.task_id])
    assert "returns 422 on invalid input" in owner.acceptance_criteria


def test_validate_merge_accepts_good_plan():
    assert validate_merge(make_spec(11, 4), good_plan(), 8) == []


def _bad_cases():
    t = ids(11)
    nine = [["T1", "T2"], ["T3", "T4"]] + [[x] for x in t[4:]]
    dup = [g.source_ids[:] for g in good_plan().groups]
    dup[1] = ["T3", "T3"]
    missing = [g.source_ids[:] for g in good_plan().groups]
    missing[0] = ["T1"]  # T2 dropped
    unknown = [g.source_ids[:] for g in good_plan().groups]
    unknown[3] = ["T99"]
    noncontig = [["T1", "T11"], ["T2", "T3"], ["T4", "T5"], ["T6", "T7"], ["T8"], ["T9"], ["T10"]]
    order = [["T3", "T4"], ["T1", "T2"]] + [[x] for x in t[4:]]
    reversed_ = [["T1", "T2"], ["T4", "T3"]] + [[x] for x in t[4:]]
    return [
        pytest.param(nine, "9 groups", id="nine-groups"),
        pytest.param(dup, "T3", id="duplicate"),
        pytest.param(missing, "T2", id="missing"),
        pytest.param(unknown, "T99", id="unknown"),
        pytest.param(noncontig, "T11", id="non-contiguous"),
        pytest.param(order, "T3", id="out-of-order"),
        pytest.param(reversed_, "T4", id="reversed"),
    ]


@pytest.mark.parametrize("groups,culprit", _bad_cases())
def test_validate_merge_flags(groups, culprit):
    problems = validate_merge(make_spec(11, 4), plan_of(*groups), 8)
    assert problems
    assert any(culprit in p for p in problems)


def test_consolidate_uses_valid_llm_plan(settings):
    model = ScriptedChatModel(responses=[structured("MergePlan", good_plan().model_dump())])
    tasks, cons = PMAgent(settings, model=model).consolidate(make_spec(11, 4))
    assert cons.method == "llm"
    assert len(tasks) <= 8
    assert [t.title for t in tasks] == [f"G{i}" for i in range(1, 9)]


def test_consolidate_falls_back_on_invalid_plan(settings):
    t = ids(11)
    bad = plan_of(["T1", "T2"], ["T3", "T3"], t[3:6], t[6:8], t[8:9], t[9:10], t[10:11])
    model = ScriptedChatModel(responses=[structured("MergePlan", bad.model_dump())])
    tasks, cons = PMAgent(settings, model=model).consolidate(make_spec(11, 4))
    assert cons.method == "deterministic"
    assert len(tasks) == 8


def test_consolidate_falls_back_on_model_error(settings):
    model = ScriptedChatModel(responses=[RuntimeError("boom")])
    tasks, cons = PMAgent(settings, model=model).consolidate(make_spec(11, 4))
    assert cons.method == "deterministic"
    assert len(tasks) == 8


def test_consolidation_records_originals(settings):
    spec = make_spec(11, 4)
    model = ScriptedChatModel(responses=[RuntimeError("boom")])
    _, cons = PMAgent(settings, model=model).consolidate(spec)
    assert cons.original_tasks == spec.tasks

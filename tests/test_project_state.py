import json

import pytest
from pydantic import ValidationError

from agents.result import CoderResult, ExecutionResult
from orchestration.project_state import (
    ProjectState,
    Spec,
    Task,
    TaskArtifact,
    TaskHandoff,
    TaskRecord,
    initial_project_state,
    state_to_jsonable,
)
from tests.fakes import make_spec


def _task(**over):
    base = dict(task_id="T1", title="t", description="d", acceptance_criteria=["a"], priority="high", covers=["AC1"])
    return Task(**{**base, **over})


def test_valid_spec_round_trips():
    s = make_spec()
    assert Spec.model_validate_json(s.model_dump_json()) == s
    assert s.criterion_ids() == ["AC1", "AC2"]


def test_duplicate_task_ids_rejected():
    s = make_spec()
    data = s.model_dump()
    data["tasks"][1]["task_id"] = "T1"
    with pytest.raises(ValidationError):
        Spec.model_validate(data)


def test_duplicate_criterion_ids_rejected():
    data = make_spec().model_dump()
    data["acceptance_criteria"][1]["id"] = "AC1"
    with pytest.raises(ValidationError):
        Spec.model_validate(data)


def test_unknown_covers_id_rejected():
    data = make_spec().model_dump()
    data["tasks"][1]["covers"] = ["AC9"]
    with pytest.raises(ValidationError, match="AC9"):
        Spec.model_validate(data)


def test_uncovered_criterion_rejected():
    data = make_spec().model_dump()
    data["tasks"][0]["covers"] = ["AC1"]
    data["tasks"][1]["covers"] = ["AC1"]
    with pytest.raises(ValidationError, match="AC2"):
        Spec.model_validate(data)


def test_extra_field_rejected():
    with pytest.raises(ValidationError):
        _task(owner="x")


def test_bad_priority_rejected():
    with pytest.raises(ValidationError):
        _task(priority="urgent")


def test_empty_lists_rejected():
    with pytest.raises(ValidationError):
        _task(acceptance_criteria=[])
    with pytest.raises(ValidationError):
        _task(covers=[])


def test_handoff_json_round_trip():
    h = TaskHandoff(
        task_id="T1", title="t", description="d", acceptance_criteria=["a"], priority="high",
        position=1, total=2, spec_name="Demo API", spec_description="x", prior_files=[],
    )
    assert set(json.loads(h.model_dump_json())) == {
        "task_id", "title", "description", "acceptance_criteria", "priority",
        "position", "total", "spec_name", "spec_description", "prior_files",
    }


def test_initial_state_has_all_fields():
    st = initial_project_state("r")
    assert set(st) == set(ProjectState.__annotations__)
    assert st["requirement"] == "r" and st["status"] == "running" and st["current_task_index"] == 0


def test_state_to_jsonable_dumps_models():
    spec = make_spec()
    art = TaskArtifact(
        task_id="T1",
        result=CoderResult(
            code="", explanation="boom", plan=["p"], result=ExecutionResult(executed=False, target=None, exit_code=None),
            status="failed", iterations=1, files={}, memories_used=[], run_id="r1", verified=False,
        ),
    )
    st = initial_project_state("r")
    st.update(spec=spec, task_status={"T1": TaskRecord(task_id="T1", status="failed", coder_status="failed")}, artifacts=[art])
    out = state_to_jsonable(st)
    json.dumps(out)
    assert Spec.model_validate(out["spec"]) == spec
    assert out["task_status"]["T1"]["status"] == "failed"
    assert out["artifacts"][0]["result"]["status"] == "failed"


@pytest.mark.parametrize("task_id", ["feature/health", "../T1", "t1", "T0", "Task1", "M1", ""])
def test_spec_rejects_unsupported_task_ids(task_id):
    data = make_spec().model_dump()
    data["tasks"][0]["task_id"] = task_id
    with pytest.raises(ValidationError, match="task id"):
        Spec.model_validate(data)

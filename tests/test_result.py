import pytest
from pydantic import ValidationError

from agents.result import CoderResult, ExecutionResult

OK = ExecutionResult(executed=True, target="t.py", exit_code=0)


def make(**kw):
    base = dict(code="x", explanation="e", plan=["p"], result=OK, status="success", iterations=3,
                files={"t.py": "x"}, memories_used=[], run_id="r", verified=True)
    return CoderResult(**(base | kw))


def test_valid_success():
    assert make().status == "success"


@pytest.mark.parametrize("kw", [
    {"verified": False},
    {"result": ExecutionResult(executed=False, target=None, exit_code=None)},
    {"result": ExecutionResult(executed=True, target="t.py", exit_code=1)},
    {"result": ExecutionResult(executed=True, target="t.py", exit_code=None, timed_out=True)},
])
def test_success_requires_real_verified_run(kw):
    with pytest.raises(ValidationError, match="success requires"):
        make(**kw)


def test_failed_may_be_unverified():
    assert make(status="failed", verified=False).status == "failed"


def test_iterations_capped():
    with pytest.raises(ValidationError):
        make(iterations=11)


def test_required_fields_present():
    assert {"code", "explanation", "plan", "result"} <= set(CoderResult.model_fields)

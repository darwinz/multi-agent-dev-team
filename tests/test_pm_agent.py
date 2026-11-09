"""PMAgent.draft_spec: structured output, one repair retry, error wrapping."""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from agents.pm_agent import PMAgent, PMError
from agents.prompts import PM_SYSTEM_PROMPT
from tests.fakes import ScriptedChatModel, make_spec, spec_response, structured


def pm(settings, *responses):
    model = ScriptedChatModel(responses=list(responses))
    return PMAgent(settings, model=model), model


def invalid_payload() -> dict:
    """Valid shape, but AC2 is covered by no task."""
    payload = make_spec().model_dump()
    for task in payload["tasks"]:
        task["covers"] = ["AC1"]
    return payload


def test_draft_spec_returns_spec(settings):
    agent, _ = pm(settings, spec_response(make_spec()))
    assert agent.draft_spec("Build a REST API") == make_spec()


def test_prompt_carries_requirement_and_rules(settings):
    agent, model = pm(settings, spec_response(make_spec()))
    agent.draft_spec("Build a REST API")
    first = model.calls[0]
    assert isinstance(first[0], SystemMessage) and first[0].content == PM_SYSTEM_PROMPT
    humans = [m for m in first if isinstance(m, HumanMessage)]
    assert "Build a REST API" in humans[0].content


def test_prompt_omits_task_cap():
    assert "8" not in PM_SYSTEM_PROMPT
    assert "eight" not in PM_SYSTEM_PROMPT.lower()


def test_invalid_then_valid_retries_once(settings):
    agent, model = pm(settings, structured("Spec", invalid_payload()), spec_response(make_spec()))
    assert agent.draft_spec("Build a REST API") == make_spec()
    assert len(model.calls) == 2
    last_human = [m for m in model.calls[1] if isinstance(m, HumanMessage)][-1]
    assert "AC2" in last_human.content


def test_two_invalid_raises(settings):
    agent, model = pm(settings, structured("Spec", invalid_payload()), structured("Spec", invalid_payload()))
    with pytest.raises(PMError):
        agent.draft_spec("Build a REST API")
    assert len(model.calls) == 2


def test_model_exception_raises_pm_error(settings):
    agent, _ = pm(settings, RuntimeError("boom"))
    with pytest.raises(PMError, match="boom"):
        agent.draft_spec("Build a REST API")


def test_unsafe_task_id_is_repaired(settings):
    # Task ids name evidence files (handoff-<id>.json); a schema-valid "feature/health" must be
    # rejected as invalid so the repair retry fixes it, not crash the run while writing evidence
    bad = make_spec().model_dump()
    bad["tasks"][1]["task_id"] = "feature/health"
    agent, model = pm(settings, structured("Spec", bad), spec_response(make_spec()))
    assert agent.draft_spec("Build a REST API") == make_spec()
    assert len(model.calls) == 2
    last_human = [m for m in model.calls[1] if isinstance(m, HumanMessage)][-1]
    assert "feature/health" in last_human.content


def test_model_init_error_raises_pm_error(settings, monkeypatch):
    # A missing provider package fails inside make_model, before any call
    def no_provider(_settings, _role):
        raise ImportError("langchain_openai is not installed")

    monkeypatch.setattr("agents.pm_agent.make_model", no_provider)
    with pytest.raises(PMError, match="langchain_openai"):
        PMAgent(settings).draft_spec("Build a REST API")


def test_structured_output_unsupported_raises_pm_error(settings):
    class NoStructured(ScriptedChatModel):
        def bind_tools(self, *a, **kw):  # the wrapper's with_structured_output binds the schema as a tool
            raise NotImplementedError("structured output not supported")

    with pytest.raises(PMError, match="not supported"):
        PMAgent(settings, model=NoStructured(responses=[])).draft_spec("Build a REST API")


def test_prompt_limits_criteria_to_stdlib_and_pytest():
    assert "standard library and pytest" in PM_SYSTEM_PROMPT

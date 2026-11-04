"""Coder self-critique and the revision/fix task text."""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from agents.coder_agent import ReflectionError
from orchestration.a2a import FixInstruction
from orchestration.handoff import render_fix, render_revision
from orchestration.project_state import TaskHandoff
from orchestration.review import Critique, CritiqueItem, Issue
from tests.fakes import PAL, PAL_REVISED, agent, critique_response, task_scripts

TASK = "Task T1: write is_pal"


def _handoff() -> TaskHandoff:
    return TaskHandoff(task_id="T1", title="Palindrome", description="Write is_pal.",
                       acceptance_criteria=["detects palindromes"], priority="high", position=1,
                       total=1, spec_name="Demo", spec_description="d", prior_files=[])


def test_critique_returns_items(settings):
    a = agent(settings, critique_response())
    out = a.critique(TASK, {"palindrome.py": PAL})
    assert isinstance(out, Critique)
    assert out.items[0].kind == "edge_case"


def test_critique_prompt_sees_task_and_files(settings):
    a = agent(settings, critique_response())
    a.critique(TASK, {"palindrome.py": PAL})
    text = "\n".join(str(m.content) for m in a.model.calls[0])
    assert TASK in text and PAL in text and "palindrome.py" in text


def test_critique_retries_then_raises(settings):
    bad = critique_response()
    bad.tool_calls[0]["args"] = {"items": []}  # violates min_length=1
    a = agent(settings, bad, bad)
    with pytest.raises(ReflectionError):
        a.critique(TASK, {"palindrome.py": PAL})
    assert len(a.model.calls) == 2
    assert "invalid" in [m for m in a.model.calls[1] if isinstance(m, HumanMessage)][-1].content


def test_critique_retry_can_succeed(settings):
    bad = critique_response()
    bad.tool_calls[0]["args"] = {"items": []}
    a = agent(settings, bad, critique_response())
    assert a.critique(TASK, {"palindrome.py": PAL}).items


def test_render_revision_lists_every_item():
    c = Critique(items=[
        CritiqueItem(kind="bug", location="palindrome.py:2", concern="case matters", change="lower() first"),
        CritiqueItem(kind="edge_case", location="palindrome.py", concern="empty string", change="add a test"),
    ])
    text = render_revision(_handoff(), c)
    assert "Task T1" in text and "Self-review" in text
    assert "- [bug] palindrome.py:2: case matters → lower() first" in text
    assert "- [edge_case] palindrome.py: empty string → add a test" in text
    assert "run each test file with exec_python before finishing" in text


def test_render_fix_names_failure_fix_and_owned_file():
    fix = FixInstruction(task_id="T1", round=2, summary="s", issues=[
        Issue(severity="High", test="test_pal_qa", criterion="detects palindromes", location="palindrome.py:2",
              assertion="assert not is_pal('ab')", fix="return False for non-palindromes")])
    text = render_fix(_handoff(), fix)
    assert "QA review round 2 failed:" in text
    assert "test_pal_qa (detects palindromes): assert not is_pal('ab') → fix: return False for non-palindromes" in text
    assert "test_qa_t1.py" in text and "do not modify" in text


def test_task_scripts_revision_differs():
    coder, qa = task_scripts()
    assert PAL_REVISED != PAL and len(qa) == 1 and len(coder) == 5

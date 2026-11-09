"""QAAgent: writes a pytest suite from the criteria, runs it for real against mocked code, reports."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from agents.config import Settings
from agents.prompts import QA_TESTS_PROMPT
from agents.qa_agent import QAAgent, QAError, canonical_suite_path
from orchestration.a2a import ReviewRequest
from orchestration.project_state import Task, TaskHandoff
from orchestration.review import QAReport, QATestSuite, TestCorrection
from tests.fakes import ScriptedChatModel, audit_response, fix_plan_response, qa_agent, qa_suite_response
from tools.workspace import ToolError, sha256_file

CRITERIA = ["push adds an item that pop can return", "pop returns the most recently pushed item"]

# Mocked code input: the tests write these files themselves, no Coder involved
BUGGY_STACK = """\
class Stack:
    def __init__(self):
        self._items = []

    def push(self, item):
        self._items.append(item)

    def peek(self):
        return self._items[-1]

    def pop(self):
        top = self.peek()
        item = self._items.pop(0)  # bug: removes the first item, not the top
        assert item == top, f"popped {item!r} but the top was {top!r}"
        return item
"""

FIXED_STACK = BUGGY_STACK.replace("self._items.pop(0)", "self._items.pop()")

HANGING_STACK = """\
class Stack:
    def push(self, item):
        while True:
            pass
"""

STACK_TESTS = """\
from stack import Stack


def test_push_then_pop():
    s = Stack()
    s.push(1)
    assert s.pop() == 1


def test_pop_returns_last():
    s = Stack()
    s.push(1)
    s.push(2)
    assert s.pop() == 2
"""
STACK_MAP = (("test_push_then_pop", 1), ("test_pop_returns_last", 2))

COUNTER = "def count(xs):\n    return len(xs)\n"
COUNTER_TESTS = "from counter import count\n\n\ndef test_count():\n    assert count([1, 2]) == 2\n"


def handoff(task_id: str = "T1", criteria: list[str] = CRITERIA) -> TaskHandoff:
    return TaskHandoff(
        task_id=task_id, title="Stack", description="A LIFO stack.", acceptance_criteria=list(criteria),
        priority="high", position=1, total=1, spec_name="Demo", spec_description="A demo.", prior_files=[],
    )


def put(settings: Settings, **files: str) -> dict[str, str]:
    """Write files into the workspace; returns {path: sha256} for a review request."""
    ws = settings.workspace_dir
    ws.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (ws / name).write_text(content)
    return {name: sha256_file(ws / name) for name in files}


def request(files: dict[str, str], task_id: str = "T1", round: int = 1) -> ReviewRequest:
    return ReviewRequest(task_id=task_id, round=round, files=files)


def stack_suite() -> QATestSuite:
    return QATestSuite.model_validate(qa_suite_response(STACK_TESTS, STACK_MAP).tool_calls[0]["args"])


def approved_t1(settings: Settings) -> QATestSuite:
    """Round 1 of T1 against the fixed stack: approved, canonical suite saved."""
    files = put(settings, **{"stack.py": FIXED_STACK})
    report, suite = qa_agent(settings, qa_suite_response(STACK_TESTS, STACK_MAP)).review(
        handoff(), request(files), 1, None, {})
    assert report.approved, report.issues
    return suite


def test_mocked_code_produces_issue_schema(settings):
    files = put(settings, **{"stack.py": BUGGY_STACK})
    agent = qa_agent(settings, qa_suite_response(STACK_TESTS, STACK_MAP), audit_response(),
                     fix_plan_response(("test_pop_returns_last", "pop removes index 0", "Use self._items.pop()")))
    report, suite = agent.review(handoff(), request(files), 1, None, {})

    assert report.approved is False
    assert (report.tests_run, report.passed, report.failed) == (2, 1, 1)
    assert QAReport.model_validate(report.model_dump()) == report
    issue = report.issues[0]
    assert issue.severity == "High"
    assert issue.test == "test_pop_returns_last"
    assert issue.criterion == CRITERIA[1]
    assert "assert" in issue.assertion
    assert issue.location.startswith("stack.py:")
    assert issue.fix == "Use self._items.pop()"
    assert (settings.workspace_dir / "test_qa_t1.py").read_text() == suite.content
    assert report.tested_files == files
    assert report.test_file == "test_qa_t1.py" and report.suites_run == {"T1": report.suite_sha}


def test_passing_code_is_approved(settings):
    files = put(settings, **{"stack.py": FIXED_STACK})
    # One scripted response: a fix-plan call would exhaust the script and fail the review
    report, _ = qa_agent(settings, qa_suite_response(STACK_TESTS, STACK_MAP)).review(
        handoff(), request(files), 1, None, {})
    assert report.approved is True
    assert (report.tests_run, report.passed, report.exit_code, report.issues) == (2, 2, 0, [])
    assert report.tampered is False


def test_hanging_code_times_out(tmp_path):
    settings = Settings.from_env(project_dir=tmp_path, exec_timeout_s=2)
    files = put(settings, **{"stack.py": HANGING_STACK})
    suite = "from stack import Stack\n\n\ndef test_push():\n    Stack().push(1)\n"
    report, _ = qa_agent(settings, qa_suite_response(suite, (("test_push", 1),))).review(
        handoff(criteria=CRITERIA[:1]), request(files), 1, None, {})
    assert report.timed_out is True and report.approved is False
    (high,) = [i for i in report.issues if i.severity == "High"]  # no per-test "did not run" noise
    assert "timed out" in high.assertion
    assert report.missing == ["test_push"]  # still recorded, so approval stays blocked
    assert report.duration_s < 6


def test_collection_error_is_issue(settings):
    files = put(settings, **{"stack.py": FIXED_STACK})
    suite = "from stack import Stack, MissingThing\n\n\ndef test_a():\n    assert Stack()\n"
    report, _ = qa_agent(settings, qa_suite_response(suite, (("test_a", 1),))).review(
        handoff(criteria=CRITERIA[:1]), request(files), 1, None, {})
    assert report.approved is False
    assert any(i.severity == "High" and "MissingThing" in i.assertion for i in report.issues)
    assert report.missing == ["test_a"]


def test_invalid_suite_retried_once(settings):
    files = put(settings, **{"stack.py": FIXED_STACK})
    bad = qa_suite_response(STACK_TESTS, (("test_push_then_pop", 1), ("test_nope", 2)))
    agent = qa_agent(settings, bad, qa_suite_response(STACK_TESTS, STACK_MAP))
    report, suite = agent.review(handoff(), request(files), 1, None, {})
    retry = "\n".join(str(m.content) for m in agent.model.inner.calls[1])
    assert "test_nope" in retry
    assert suite is not None and report.approved is True


@pytest.mark.parametrize("second", ["invalid", "raises"])
def test_suite_failure_is_qa_error(settings, second):
    files = put(settings, **{"stack.py": FIXED_STACK})
    bad = qa_suite_response(STACK_TESTS, (("test_nope", 1), ("test_pop_returns_last", 2)))
    responses = [bad, bad] if second == "invalid" else [RuntimeError("provider down")]
    report, suite = qa_agent(settings, *responses).review(handoff(), request(files), 1, None, {})
    assert report.qa_error and suite is None
    assert report.tests_run == 0 and report.approved is False


def test_generate_suite_raises_qa_error(settings):
    agent = QAAgent(settings, model=ScriptedChatModel(responses=[RuntimeError("boom")]))
    with pytest.raises(QAError, match="boom"):
        agent.generate_suite(handoff(), {"stack.py": FIXED_STACK})


def test_suite_restored_when_tampered(settings):
    suite = approved_t1(settings)
    test_file = settings.workspace_dir / "test_qa_t1.py"
    test_file.write_text("def test_push_then_pop():\n    pass\n\n\ndef test_pop_returns_last():\n    pass\n")
    files = put(settings, **{"stack.py": BUGGY_STACK})
    report, again = qa_agent(settings).review(handoff(), request(files, round=2), 2, suite, {})
    assert report.tampered is True
    assert test_file.read_text() == suite.content and again == suite
    assert report.approved is False and report.failed == 1


def test_canonical_suite_outside_workspace(settings):
    suite = approved_t1(settings)
    canonical = canonical_suite_path(settings, "T1")
    assert canonical == settings.state_dir / "qa" / "suites" / "test_qa_t1.py"
    assert canonical.read_text() == suite.content
    assert settings.workspace_dir.resolve() not in canonical.resolve().parents


def test_all_skipped_not_approved(settings):
    files = put(settings, **{"stack.py": FIXED_STACK})
    suite = "import pytest\n\n\ndef test_later():\n    pytest.skip('not yet')\n"
    report, _ = qa_agent(settings, qa_suite_response(suite, (("test_later", 1),))).review(
        handoff(criteria=CRITERIA[:1]), request(files), 1, None, {})
    assert (report.skipped, report.exit_code, report.approved) == (1, 0, False)
    assert any(i.severity == "High" and i.test == "test_later" for i in report.issues)


def test_partially_skipped_not_approved(settings):
    files = put(settings, **{"stack.py": FIXED_STACK})
    suite = STACK_TESTS.replace("    s.push(2)\n", "    import pytest\n    pytest.skip('later')\n")
    report, _ = qa_agent(settings, qa_suite_response(suite, STACK_MAP)).review(
        handoff(), request(files), 1, None, {})
    assert (report.passed, report.skipped, report.approved) == (1, 1, False)


def test_missing_manifest_test_not_approved(settings):
    files = put(settings, **{"stack.py": FIXED_STACK})
    # test_b sits inside `if False:`, so it never runs; such a suite fails generation's check, so pass it in
    content = "def test_a():\n    assert True\n\n\nif False:\n    def test_b():\n        assert True\n"
    suite = QATestSuite(content=content, tests=[{"name": "test_a", "criterion": 1}, {"name": "test_b", "criterion": 2}])
    report, _ = qa_agent(settings).review(handoff(), request(files), 1, suite, {})
    assert report.missing == ["test_b"] and report.approved is False
    assert any(i.severity == "High" and i.test == "test_b" for i in report.issues)


def t2_review(settings: Settings, prior: dict[str, QATestSuite], *responses) -> QAReport:
    files = put(settings, **{"counter.py": COUNTER})
    agent = qa_agent(settings, qa_suite_response(COUNTER_TESTS, (("test_count", 1),)), *responses)
    report, _ = agent.review(handoff("T2", ["count returns the length"]), request(files, "T2"), 1, None, prior)
    return report


def test_prior_suite_regression_is_issue(settings):
    t1 = approved_t1(settings)
    put(settings, **{"stack.py": BUGGY_STACK})
    report = t2_review(settings, {"T1": t1})
    assert set(report.suites_run) == {"T1", "T2"}
    regressions = [i for i in report.issues if i.assertion.startswith("regression in T1: ")]
    assert [i.test for i in regressions] == ["test_pop_returns_last"]
    assert regressions[0].severity == "High" and report.approved is False


def test_prior_suite_restored_when_deleted(settings):
    t1 = approved_t1(settings)
    (settings.workspace_dir / "test_qa_t1.py").unlink()
    report = t2_review(settings, {"T1": t1})
    assert (settings.workspace_dir / "test_qa_t1.py").read_text() == t1.content
    assert report.tampered is True
    assert report.tests_run == 3 and report.approved is True


def test_stale_file_is_issue(settings):
    put(settings, **{"stack.py": FIXED_STACK})
    report, _ = qa_agent(settings, qa_suite_response(STACK_TESTS, STACK_MAP)).review(
        handoff(), request({"stack.py": "0" * 64}), 1, None, {})
    stale = [i for i in report.issues if "changed after review request" in i.assertion]
    assert stale and stale[0].severity == "High"
    assert report.tested_files == {"stack.py": sha256_file(settings.workspace_dir / "stack.py")}
    assert report.passed == 2 and report.approved is False


def test_fix_plan_failure_falls_back(settings):
    files = put(settings, **{"stack.py": BUGGY_STACK})
    agent = qa_agent(settings, qa_suite_response(STACK_TESTS, STACK_MAP), RuntimeError("fix plan down"))
    report, _ = agent.review(handoff(), request(files), 1, None, {})
    assert report.issues[0].fix.startswith("Make test_pop_returns_last pass: ")


def test_run_suites_never_calls_model(settings):
    t1 = approved_t1(settings)
    put(settings, **{"stack.py": BUGGY_STACK})
    report = qa_agent(settings).run_suites({"T1": t1}, task_id="regression", round=0)
    assert report.failed == 1 and report.approved is False
    assert report.issues[0].assertion.startswith("regression in T1: ")


def test_qa_records_outside_workspace(settings):
    approved_t1(settings)
    assert list(Path(settings.state_dir / "qa").glob("*.xml"))
    assert not list(settings.workspace_dir.glob("*.xml"))


def test_round_one_replaces_stale_canonical(settings):
    canonical = canonical_suite_path(settings, "T1")
    canonical.parent.mkdir(parents=True)
    canonical.write_text("def test_old():\n    assert False\n")  # left by an earlier run
    suite = approved_t1(settings)
    assert canonical.read_text() == suite.content


UNMAPPED_EXTRA = STACK_TESTS + "\n\ndef test_extra():\n    assert True\n"


def test_unmapped_extra_test_retried(settings):
    files = put(settings, **{"stack.py": FIXED_STACK})
    agent = qa_agent(settings, qa_suite_response(UNMAPPED_EXTRA, STACK_MAP), qa_suite_response(STACK_TESTS, STACK_MAP))
    report, suite = agent.review(handoff(), request(files), 1, None, {})
    assert "test_extra" in "\n".join(str(m.content) for m in agent.model.inner.calls[1])
    assert suite.content == STACK_TESTS and report.approved is True


def test_unmapped_extra_test_twice_is_qa_error(settings):
    files = put(settings, **{"stack.py": FIXED_STACK})
    bad = qa_suite_response(UNMAPPED_EXTRA, STACK_MAP)
    report, suite = qa_agent(settings, bad, bad).review(handoff(), request(files), 1, None, {})
    assert report.qa_error and "test_extra" in report.qa_error and suite is None
    assert report.tests_run == 0 and report.approved is False


def test_long_stale_path_does_not_raise(settings):
    put(settings, **{"stack.py": FIXED_STACK})
    long_path = "d" * 300 + "/" + "x" * 300 + ".py"
    report, _ = qa_agent(settings, qa_suite_response(STACK_TESTS, STACK_MAP)).review(
        handoff(), request({"stack.py": sha256_file(settings.workspace_dir / "stack.py"), long_path: "0" * 64}),
        1, None, {})
    stale = [i for i in report.issues if i.location == long_path]
    assert stale and len(stale[0].assertion) <= 500 and report.approved is False


def test_collection_error_does_not_hide_prior_suites(settings):
    t1 = approved_t1(settings)
    files = put(settings, **{"counter.py": COUNTER})
    broken = "from counter import count, MissingThing\n\n\ndef test_count():\n    assert count([1]) == 1\n"
    agent = qa_agent(settings, qa_suite_response(broken, (("test_count", 1),)))
    report, _ = agent.review(handoff("T2", ["count returns the length"]), request(files, "T2"), 1, None, {"T1": t1})
    t1_results = {r.name: r.outcome for r in report.results if r.name.startswith("test_qa_t1::")}
    assert t1_results == {"test_qa_t1::test_push_then_pop": "passed", "test_qa_t1::test_pop_returns_last": "passed"}
    assert not [i for i in report.issues if i.assertion.startswith("regression in T1")]
    assert any(i.severity == "High" and "MissingThing" in i.assertion for i in report.issues)
    assert report.missing == ["test_count"] and report.approved is False


def test_canonical_suite_tampering_detected(settings):
    suite = approved_t1(settings)
    canonical = canonical_suite_path(settings, "T1")
    canonical.write_text("def test_push_then_pop():\n    pass\n\n\ndef test_pop_returns_last():\n    pass\n")
    files = put(settings, **{"stack.py": BUGGY_STACK})
    report, _ = qa_agent(settings).review(handoff(), request(files, round=2), 2, suite, {})
    assert report.tampered is True
    assert report.failed == 1 and report.approved is False  # the state-held suite ran, not the rewrite
    assert canonical.read_text() == suite.content
    assert (settings.workspace_dir / "test_qa_t1.py").read_text() == suite.content


def test_canonical_suite_deleted_is_tampering(settings):
    suite = approved_t1(settings)
    canonical_suite_path(settings, "T1").unlink()
    report = qa_agent(settings).run_suites({"T1": suite}, task_id="regression", round=0)
    assert report.tampered is True and report.approved is True
    assert canonical_suite_path(settings, "T1").read_text() == suite.content


def test_workspace_addopts_ignored(settings):
    files = put(settings, **{"stack.py": FIXED_STACK, "pytest.ini": "[pytest]\naddopts = -k nothing\n"})
    report, _ = qa_agent(settings, qa_suite_response(STACK_TESTS, STACK_MAP)).review(
        handoff(), request(files), 1, None, {})
    assert (report.tests_run, report.passed, report.missing) == (2, 2, [])
    assert report.approved is True, report.issues


def later_task(task_id: str = "T2") -> Task:
    return Task(task_id=task_id, title="Insert at index", description="insert(value, index).",
                acceptance_criteria=["insert(x, 0) on a non-empty list puts x first"], priority="high",
                covers=["AC1"])


def suite_prompt(agent: QAAgent) -> str:
    return "\n".join(m.content for m in agent.model.inner.calls[0] if isinstance(m, HumanMessage))


def test_suite_prompt_lists_later_tasks(settings):
    files = put(settings, **{"stack.py": FIXED_STACK})
    agent = qa_agent(settings, qa_suite_response(STACK_TESTS, STACK_MAP))
    agent.review(handoff(), request(files), 1, None, {}, later=[later_task()])
    prompt = suite_prompt(agent)
    assert "Later tasks" in prompt
    assert "T2: Insert at index" in prompt
    assert "insert(x, 0) on a non-empty list puts x first" in prompt


def test_suite_prompt_without_later_tasks(settings):
    files = put(settings, **{"stack.py": FIXED_STACK})
    agent = qa_agent(settings, qa_suite_response(STACK_TESTS, STACK_MAP))
    agent.review(handoff(), request(files), 1, None, {})
    assert "Later tasks" not in suite_prompt(agent)


def test_tests_prompt_scopes_suite_to_task():
    assert "later task" in QA_TESTS_PROMPT
    assert "NotImplementedError" in QA_TESTS_PROMPT
    assert "standard library" in QA_TESTS_PROMPT


# A suite whose second test miscomputes its expected value: correct code can never pass it
WRONG_TEST = """\
def test_pop_returns_last():
    s = Stack()
    s.push(1)
    s.push(2)
    assert s.pop() == 1
"""
RIGHT_TEST = """\
def test_pop_returns_last():
    s = Stack()
    s.push(1)
    s.push(2)
    assert s.pop() == 2
"""
FIRST_TEST = STACK_TESTS[:STACK_TESTS.index("def test_pop_returns_last")]
WRONG_SUITE = FIRST_TEST + WRONG_TEST


def test_wrong_test_corrected_in_round_one(settings):
    files = put(settings, **{"stack.py": FIXED_STACK})
    agent = qa_agent(settings, qa_suite_response(WRONG_SUITE, STACK_MAP),
                     audit_response(("test_pop_returns_last", True, "pushing 1 then 2 pops 2, not 1", RIGHT_TEST)))
    report, suite = agent.review(handoff(), request(files), 1, None, {})
    assert report.approved, report.issues
    assert report.test_corrections == [TestCorrection(test="test_pop_returns_last",
                                                      reason="pushing 1 then 2 pops 2, not 1")]
    assert suite.content == FIRST_TEST + RIGHT_TEST
    assert canonical_suite_path(settings, "T1").read_text() == suite.content
    assert (settings.workspace_dir / "test_qa_t1.py").read_text() == suite.content


def test_correct_test_kept_and_fix_planned(settings):
    files = put(settings, **{"stack.py": BUGGY_STACK})
    agent = qa_agent(settings, qa_suite_response(STACK_TESTS, STACK_MAP),
                     audit_response(("test_pop_returns_last", False, "the code pops the first item", "")),
                     fix_plan_response(("test_pop_returns_last", "pop removes index 0", "Use self._items.pop()")))
    report, suite = agent.review(handoff(), request(files), 1, None, {})
    assert report.approved is False and report.test_corrections == []
    assert suite.content == STACK_TESTS
    assert report.issues[0].fix == "Use self._items.pop()"


@pytest.mark.parametrize("corrected", [
    RIGHT_TEST.replace("test_pop_returns_last", "test_other"),  # renamed
    RIGHT_TEST + "\n\ndef test_extra():\n    pass\n",  # adds a test
    "def test_pop_returns_last(:\n",  # does not parse
])
def test_invalid_correction_rejected(settings, corrected):
    files = put(settings, **{"stack.py": FIXED_STACK})
    agent = qa_agent(settings, qa_suite_response(WRONG_SUITE, STACK_MAP),
                     audit_response(("test_pop_returns_last", True, "wrong expectation", corrected)),
                     fix_plan_response())
    report, suite = agent.review(handoff(), request(files), 1, None, {})
    assert report.approved is False and report.test_corrections == []
    assert suite.content == WRONG_SUITE


def test_no_test_check_after_round_one(settings):
    suite = QATestSuite.model_validate(qa_suite_response(WRONG_SUITE, STACK_MAP).tool_calls[0]["args"])
    files = put(settings, **{"stack.py": FIXED_STACK})
    agent = qa_agent(settings, fix_plan_response())
    report, again = agent.review(handoff(), request(files, round=2), 2, suite, {})
    assert len(agent.model.inner.calls) == 1  # the fix plan only
    assert report.approved is False and again == suite


def test_qa_direct_tool_args_validated(settings, monkeypatch):
    """A suite file path that escapes the workspace is rejected before write_file runs (spec §5.7);
    the error takes QA's existing crash path, which reports qa_error."""
    from orchestration.team_graph import _qa_crash

    files = put(settings, **{"stack.py": FIXED_STACK})
    writes = []
    monkeypatch.setattr("agents.qa_agent.qa_test_file", lambda task_id: "../escaped_test_qa.py")
    monkeypatch.setattr("agents.qa_agent.write_file", lambda *a, **kw: writes.append(a))
    agent = qa_agent(settings, qa_suite_response(STACK_TESTS, STACK_MAP))
    with pytest.raises(ToolError, match="outside the workspace") as raised:
        agent.review(handoff(), request(files), 1, None, {})
    assert writes == []
    assert not (settings.workspace_dir.parent / "escaped_test_qa.py").exists()
    report = _qa_crash("T1", 1, raised.value)
    assert report.qa_error.startswith("QA crashed: WorkspaceError") and report.approved is False


def test_qa_pytest_args_validated(settings, monkeypatch):
    agent = qa_agent(settings.model_copy(update={"exec_timeout_s": 0}))  # not a usable exec timeout
    ran = []
    monkeypatch.setattr("agents.qa_agent.exec_python", lambda *a, **kw: ran.append(kw))
    with pytest.raises(ToolError, match="timeout"):
        agent._pytest(["test_qa_t1.py"], settings.project_dir / "j.xml")
    assert ran == []

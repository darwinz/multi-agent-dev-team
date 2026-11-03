"""QAAgent: writes a pytest suite from the criteria, runs it for real against mocked code, reports."""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.config import Settings
from agents.qa_agent import QAAgent, QAError, canonical_suite_path
from orchestration.a2a import ReviewRequest
from orchestration.project_state import TaskHandoff
from orchestration.review import QAReport, QATestSuite
from tests.fakes import ScriptedChatModel, fix_plan_response, qa_agent, qa_suite_response
from tools.workspace import sha256_file

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
    agent = qa_agent(settings, qa_suite_response(STACK_TESTS, STACK_MAP),
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
    assert any(i.severity == "High" and "timed out" in i.assertion for i in report.issues)
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
    retry = "\n".join(str(m.content) for m in agent.model.calls[1])
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

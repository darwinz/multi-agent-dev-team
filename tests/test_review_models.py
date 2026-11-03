import pytest
from pydantic import ValidationError

from orchestration.a2a import FinalReport
from orchestration.review import (
    Critique,
    CritiqueItem,
    Issue,
    QAReport,
    QATestSuite,
    SuiteTest,
)


def issue(severity="Low"):
    return Issue(severity=severity, test="test_a", criterion="c1", location="test_a", assertion="boom", fix="fix it")


def report(**over):
    base = dict(
        task_id="T1", round=1, test_file="test_qa_t1.py", suite_sha="s", suites_run={"T1": "s"},
        tested_files={"a.py": "h"}, tests_run=2, passed=2, failed=0, errors=0, skipped=0,
        missing=[], exit_code=0, timed_out=False, duration_s=0.1, results=[], issues=[],
        tampered=False, qa_error=None, approved=True,
    )
    base.update(over)
    return QAReport(**base)


def test_clean_run_can_be_approved():
    assert report().approved is True


@pytest.mark.parametrize("over", [
    dict(tests_run=0, passed=0),
    dict(failed=1, passed=1),
    dict(tests_run=1, passed=0, skipped=1),
    dict(errors=1, passed=1),
    dict(missing=["test_x"]),
    dict(passed=1),
    dict(exit_code=1),
    dict(exit_code=None),
    dict(timed_out=True),
    dict(qa_error="x"),
    dict(issues=[issue("High")]),
])
def test_approved_requires_clean_run(over):
    with pytest.raises(ValidationError):
        report(**over)


def test_unapproved_report_allows_issues():
    r = report(approved=False, failed=1, passed=1, exit_code=1, issues=[issue("High")])
    assert r.approved is False and r.issues[0].severity == "High"


def test_approved_allows_low_issue():
    assert report(issues=[issue("Low")]).approved is True


def test_final_report_reasons():
    for reason in ("max_iterations", "qa_error", "no_revision"):
        assert FinalReport(task_id="T1", round=1, reason=reason, unresolved=[], summary="s").reason == reason
    with pytest.raises(ValidationError):
        FinalReport(task_id="T1", round=1, reason="other", unresolved=[], summary="s")


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        Issue(severity="Low", test=None, criterion=None, location="l", assertion="a", fix="f", bogus=1)


def test_issue_assertion_capped():
    with pytest.raises(ValidationError):
        Issue(severity="Low", test=None, criterion=None, location="l", assertion="x" * 501, fix="f")


SRC = "def test_a():\n    assert True\n\n\ndef test_b():\n    assert True\n"


def test_valid_suite_has_no_problems():
    s = QATestSuite(content=SRC, tests=[SuiteTest(name="test_a", criterion=1), SuiteTest(name="test_b", criterion=2)])
    assert s.check(2) == []


@pytest.mark.parametrize("content,tests,word", [
    ("def test_a(:\n", [("test_a", 1), ("test_b", 2)], "syntax"),
    (SRC, [("test_a", 1), ("test_zzz", 2)], "test_zzz"),
    (SRC, [("test_a", 1), ("test_a", 2)], "duplicate"),
    (SRC, [("test_a", 1), ("test_b", 1)], "criterion 2"),
    (SRC, [("test_a", 1), ("test_b", 5)], "criterion 5"),
])
def test_suite_check_reports_problems(content, tests, word):
    s = QATestSuite(content=content, tests=[SuiteTest(name=n, criterion=c) for n, c in tests])
    problems = s.check(2)
    assert problems and any(word in p for p in problems)


def test_nested_function_is_not_a_top_level_test():
    src = "def helper():\n    def test_inner():\n        pass\n"
    s = QATestSuite(content=src, tests=[SuiteTest(name="test_inner", criterion=1)])
    assert any("test_inner" in p for p in s.check(1))


def test_critique_needs_an_item():
    item = CritiqueItem(kind="bug", location="a.py:1", concern="c", change="x")
    with pytest.raises(ValidationError):
        Critique(items=[])
    with pytest.raises(ValidationError):
        Critique(items=[item] * 9)
    assert len(Critique(items=[item] * 8).items) == 8

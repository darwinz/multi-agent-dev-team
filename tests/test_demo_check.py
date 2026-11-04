"""check_linked_list: the grading demo's three checks, judged on canonical suites and the regression run."""

from __future__ import annotations

import pytest

from agents.config import Settings
from agents.qa_agent import canonical_suite_path
from app.demo_check import check_linked_list
from orchestration.project_state import state_to_jsonable
from orchestration.review import QATestSuite, SuiteTest
from tests.fakes import qa_agent
from tests.scenarios import BUGGY_LL, FIXED_LL, QA_LL_MANIFEST, QA_LL_TESTS, linked_list_spec

TRIVIAL = "def test_ok():\n    assert True\n"

# QA_LL_TESTS without test_reverse: `reverse` is never called
NO_REVERSE = QA_LL_TESTS[: QA_LL_TESTS.index("def test_reverse")].rstrip() + "\n"


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings.from_env(project_dir=tmp_path, exec_timeout_s=15)


def suite(content: str = QA_LL_TESTS) -> QATestSuite:
    tests = [SuiteTest(name=n, criterion=c) for n, c in QA_LL_MANIFEST if f"def {n}(" in content]
    return QATestSuite(content=content, tests=tests)


def make_state(settings, code: str = FIXED_LL, content: str = QA_LL_TESTS) -> dict:
    """The final state of a one-task run whose T1 was approved on `content` against `code`.
    QA rounds and the regression check are real pytest runs over the workspace."""
    settings.workspace_dir.mkdir(parents=True, exist_ok=True)
    (settings.workspace_dir / "linked_list.py").write_text(code)
    canonical_suite_path(settings, "T1").unlink(missing_ok=True)  # QA writes it afresh on round 1
    qa = qa_agent(settings)
    s = suite(content)
    report = qa.run_suites({"T1": s}, task_id="T1", round=1)
    regression = qa.run_suites({"T1": s}, task_id="regression", round=0)
    spec = linked_list_spec()
    done = report.approved
    return state_to_jsonable({
        "status": "completed" if done else "partial",
        "tasks": spec.tasks,
        "task_status": {"T1": {"task_id": "T1", "status": "done" if done else "failed",
                               "qa_outcome": "approved" if done else "unresolved", "qa_rounds": 1}},
        "qa_reports": [report],
        "approved_suites": {"T1": s} if done else {},
        "regression": regression if done else None,
    })


def lines(report: str) -> list[str]:
    return report.splitlines()


def test_check_passes_on_approved_full_coverage(settings):
    passed, report = check_linked_list(settings.workspace_dir, make_state(settings))
    assert passed, report
    assert len(lines(report)) == 3 and all(line.startswith("PASS ") for line in lines(report))
    assert "insert, delete, search, reverse" in report


def test_check_accepts_suite_models(settings):
    # The graph holds QATestSuite / QAReport objects; the runner hands over their dumps. Both work.
    state = make_state(settings)
    state["approved_suites"] = {"T1": suite()}
    assert check_linked_list(settings.workspace_dir, state)[0]


def test_check_fails_when_a_method_untested(settings):
    passed, report = check_linked_list(settings.workspace_dir, make_state(settings, content=NO_REVERSE))
    assert not passed
    first, methods, regression = lines(report)
    assert first.startswith("PASS ") and regression.startswith("PASS ")
    assert methods.startswith("FAIL ") and "reverse" in methods.split(":", 1)[1]


def test_check_fails_when_task_not_approved(settings):
    passed, report = check_linked_list(settings.workspace_dir, make_state(settings, code=BUGGY_LL))
    assert not passed
    assert all(line.startswith("FAIL ") for line in lines(report))
    assert "T1" in lines(report)[0]


def test_check_fails_when_regression_failed(settings):
    state = make_state(settings)
    # A later change broke reverse: the finalize regression run is not approved
    (settings.workspace_dir / "linked_list.py").write_text(BUGGY_LL)
    state["regression"] = qa_agent(settings).run_suites({"T1": suite()}, task_id="regression",
                                                        round=0).model_dump(mode="json")
    assert state["regression"]["approved"] is False
    passed, report = check_linked_list(settings.workspace_dir, state)
    assert not passed
    assert lines(report)[2].startswith("FAIL ")


def test_check_fails_without_regression_or_with_partial_coverage(settings):
    state = make_state(settings)
    passed, report = check_linked_list(settings.workspace_dir, {**state, "regression": None})
    assert not passed and lines(report)[2].startswith("FAIL ")
    # A regression run that did not restore T1's suite does not count
    partial = {**state["regression"], "suites_run": {}}
    passed, report = check_linked_list(settings.workspace_dir, {**state, "regression": partial})
    assert not passed and "T1" in lines(report)[2]


def test_check_ignores_weakened_workspace_suite(settings):
    state = make_state(settings)
    copy = settings.workspace_dir / "test_qa_t1.py"
    copy.write_text(TRIVIAL)  # the Coder weakens its workspace copy
    passed, report = check_linked_list(settings.workspace_dir, state)
    assert passed, report
    assert copy.read_text() == TRIVIAL  # never read back, never re-run

    # and the reverse: a weak canonical suite fails even when the workspace copy is complete
    weak = make_state(settings, content=NO_REVERSE)
    copy.write_text(QA_LL_TESTS)
    assert not check_linked_list(settings.workspace_dir, weak)[0]

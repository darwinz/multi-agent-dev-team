"""check_linked_list: the grading demo's three checks, judged on canonical suites and the regression run.
check_bst: the BST demo's approval line, the supplied fixture run and the reported coverage line."""

from __future__ import annotations

import pytest

from agents.config import Settings
from agents.qa_agent import canonical_suite_path
from app.demo_check import check_bst, check_linked_list
from orchestration.project_state import state_to_jsonable
from orchestration.review import QATestSuite, SuiteTest
from tests.fakes import qa_agent
from tests.scenarios import (
    BST_SRC,
    BUGGY_LL,
    FIXED_LL,
    QA_BST_MANIFEST,
    QA_BST_TESTS,
    QA_LL_MANIFEST,
    QA_LL_TESTS,
    bst_spec,
    linked_list_spec,
)

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


# --- check_bst ----------------------------------------------------------------------------

# The tree is a sorted list: inorder returns insertion order, not ascending order
UNSORTED_BST = '''\
class BinarySearchTree:
    def __init__(self):
        self.items = []

    def insert(self, value):
        if value not in self.items:
            self.items.append(value)

    def search(self, value):
        return value in self.items

    def inorder(self):
        return list(self.items)
'''


def bst_state(settings, code: str = BST_SRC) -> dict:
    """The final state of a one-task BST run whose T1 was approved on QA_BST_TESTS against `code`."""
    settings.workspace_dir.mkdir(parents=True, exist_ok=True)
    (settings.workspace_dir / "bst.py").write_text(code)
    canonical_suite_path(settings, "T1").unlink(missing_ok=True)
    s = QATestSuite(content=QA_BST_TESTS, tests=[SuiteTest(name=n, criterion=c) for n, c in QA_BST_MANIFEST])
    report = qa_agent(settings).run_suites({"T1": s}, task_id="T1", round=1)
    done = report.approved
    return state_to_jsonable({
        "status": "completed" if done else "partial",
        "tasks": bst_spec().tasks,
        "task_status": {"T1": {"task_id": "T1", "status": "done" if done else "failed",
                               "qa_outcome": "approved" if done else "unresolved", "qa_rounds": 1}},
        "qa_reports": [report],
        "approved_suites": {"T1": s} if done else {},
    })


def test_bst_fixture_passes_reference_and_fails_broken(settings):
    passed, report = check_bst(settings.workspace_dir, bst_state(settings))
    assert passed, report
    approved, fixture, coverage = lines(report)
    assert approved.startswith("PASS ") and fixture.startswith("PASS supplied BST fixture")
    assert coverage.startswith("coverage of bst.py under QA's tests: ") and coverage.endswith("(reported, not gated)")
    assert coverage.split(": ", 1)[1].split("%")[0].isdigit(), coverage

    # inorder returns insertion order; QA's small suite happens to pass, the fixture does not
    (settings.workspace_dir / "bst.py").write_text(UNSORTED_BST)
    passed, report = check_bst(settings.workspace_dir, bst_state(settings, code=UNSORTED_BST))
    assert not passed
    assert lines(report)[1].startswith("FAIL supplied BST fixture")
    assert "test_inorder_sorted_random" in report


def test_bst_check_fails_without_module_and_reports_na_coverage(settings):
    settings.workspace_dir.mkdir(parents=True, exist_ok=True)
    passed, report = check_bst(settings.workspace_dir, {"tasks": [], "task_status": {}})
    assert not passed
    approved, fixture, coverage = lines(report)
    assert approved.startswith("FAIL ") and fixture.startswith("FAIL supplied BST fixture")
    assert coverage.startswith("coverage of bst.py under QA's tests: n/a (")


def test_bst_check_children_get_no_secrets(settings, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-dummy")
    settings.workspace_dir.mkdir(parents=True, exist_ok=True)
    # bst.py records which secret-looking names it can see at import, as the fixture imports it
    (settings.workspace_dir / "bst.py").write_text(
        "import os, pathlib\n"
        "pathlib.Path(__file__).with_name('seen.txt').write_text(' '.join(sorted(k for k in os.environ "
        "if k.endswith(('_KEY', '_TOKEN')))))\n" + BST_SRC)
    assert check_bst(settings.workspace_dir, {})[1].splitlines()[1].startswith("PASS")
    assert (settings.workspace_dir / "seen.txt").read_text() == ""


BROKEN_BST = UNSORTED_BST


@pytest.mark.parametrize("trick", ["shadow_pytest", "random_exit", "os_exit_on_import"])
def test_bst_fixture_exit_0_without_a_real_run_fails(settings, trick):
    """A workspace file can stand in for pytest or a module the fixture imports, or end the process with
    exit 0 before any test runs. Without a JUnit result showing the fixture's 5 tests passed, it is FAIL."""
    ws = settings.workspace_dir
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "bst.py").write_text(BROKEN_BST)
    if trick == "shadow_pytest":
        (ws / "pytest.py").write_text("import sys\nsys.exit(0)\n")
    elif trick == "random_exit":
        (ws / "random.py").write_text("raise SystemExit(0)\n")
    else:
        (ws / "bst.py").write_text("import os\nos._exit(0)\n" + BROKEN_BST)
    passed, report = check_bst(ws, {})
    fixture = lines(report)[1]
    assert not passed
    assert fixture.startswith("FAIL supplied BST fixture"), fixture

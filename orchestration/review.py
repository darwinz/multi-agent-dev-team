"""Review models: QA's findings, its generated test suite, and the Coder's self-reflection."""

from __future__ import annotations

import ast
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agents.result import Status

Severity = Literal["High", "Medium", "Low"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Issue(_Strict):
    """One QA finding: severity, failing test, criterion covered, location, failure message and suggested fix."""

    severity: Severity
    test: str | None = Field(description="pytest node name")
    criterion: str | None = Field(description="the task criterion the test covers")
    location: str = Field(description="file.py:line, or the test name")
    assertion: str = Field(max_length=500, description="the failure message")
    fix: str


class TestOutcome(_Strict):
    """Result of one pytest test (passed, failed, error or skipped) with its message."""

    __test__ = False  # not a pytest class

    name: str
    outcome: Literal["passed", "failed", "error", "skipped"]
    message: str = ""


class SuiteTest(_Strict):
    """One manifest entry: a test name and the 1-based task criterion it covers."""

    __test__ = False

    name: str
    criterion: int = Field(description="1-based index into the task's acceptance criteria")


class QATestSuite(_Strict):
    """QA's LLM output: a pytest module plus a manifest mapping each test to a criterion."""

    content: str
    tests: list[SuiteTest]

    def check(self, n_criteria: int) -> list[str]:
        """Problems with the suite in words (empty means valid). A method, not a validator,
        so the QA retry can feed the problems back to the model."""
        try:
            tree = ast.parse(self.content)
        except SyntaxError as e:
            return [f"the test module has a syntax error: {e.msg} (line {e.lineno})"]
        defined = [
            n.name for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test_")
        ]
        problems: list[str] = [
            f"class {n.name} holds test methods; tests must be plain top-level functions"
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name.startswith("Test") and any(
                isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name.startswith("test")
                for m in n.body)
        ]
        seen: set[str] = set()
        for t in self.tests:
            if t.name in seen:
                problems.append(f"duplicate test name in the manifest: {t.name}")
            seen.add(t.name)
            if t.name not in defined:
                problems.append(f"manifest test {t.name} is not a top-level def test_* in the module")
            if not 1 <= t.criterion <= n_criteria:
                problems.append(f"test {t.name} maps to criterion {t.criterion}, but criteria are 1..{n_criteria}")
        mapped = {t.name for t in self.tests}
        problems += [f"test {name} is not in the manifest; map it to a criterion or remove it"
                     for name in dict.fromkeys(defined) if name not in mapped]
        covered = {t.criterion for t in self.tests}
        problems += [f"criterion {i} has no test" for i in range(1, n_criteria + 1) if i not in covered]
        return problems


class TestCorrection(_Strict):
    """A round-1 QA test that QA judged wrong against the criteria and replaced before freezing."""

    __test__ = False

    test: str
    reason: str


class QAReport(_Strict):
    """One QA round for a task: pytest counts and outcomes, issues, tampering and approval (computed in code)."""

    task_id: str
    round: int
    test_file: str
    suite_sha: str
    suites_run: dict[str, str] = Field(description="task_id -> suite sha: this task's plus earlier approved tasks'")
    tested_files: dict[str, str]
    tests_run: int
    passed: int
    failed: int
    errors: int
    skipped: int
    missing: list[str] = Field(description="manifest tests with no passing-or-failing result")
    exit_code: int | None
    timed_out: bool
    duration_s: float
    results: list[TestOutcome]
    issues: list[Issue]
    tampered: bool
    qa_error: str | None
    test_corrections: list[TestCorrection] = Field(default_factory=list,
                                                   description="round 1 only: QA tests replaced before the rerun")
    approved: bool

    @model_validator(mode="after")
    def _approval_is_earned(self) -> QAReport:
        """Approval is computed in code from test results and never taken from a model."""
        if not self.approved:
            return self
        problems = [
            msg for ok, msg in [
                (self.tests_run >= 1, "no tests ran"),
                (self.passed == self.tests_run, "not every test passed"),
                (self.failed == 0 and self.errors == 0 and self.skipped == 0, "tests failed, errored or were skipped"),
                (not self.missing, "manifest tests are missing"),
                (self.exit_code == 0, "pytest exit code is not 0"),
                (not self.timed_out, "the run timed out"),
                (self.qa_error is None, "QA reported an error"),
                (not any(i.severity == "High" for i in self.issues), "a High issue is open"),
            ] if not ok
        ]
        if problems:
            raise ValueError("cannot approve: " + "; ".join(problems))
        return self


class CritiqueItem(_Strict):
    """One self-review finding on the Coder's own code: a bug or edge case, where, the concern and the change."""

    kind: Literal["bug", "edge_case"]
    location: str
    concern: str
    change: str


class Critique(_Strict):
    """The Coder's self-review: 1 to 8 critique items."""

    items: list[CritiqueItem] = Field(min_length=1, max_length=8)


class Reflection(_Strict):
    """A self-reflection pass: the critique, files before and after, which changed and the revision status."""

    task_id: str
    critique: Critique | None
    before: dict[str, str]
    after: dict[str, str]
    changed: list[str]
    revision_status: Status | None
    error: str | None

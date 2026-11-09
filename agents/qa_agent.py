"""QA & Debugger agent: writes a pytest suite from a task's criteria, runs it, reports and diagnoses.

Approval is computed here from real test results (spec §6–7), never taken from the model.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from agents.config import Settings
from agents.model import cut_off, make_model, resilient
from agents.prompts import QA_CHECK_PROMPT, QA_FIX_PROMPT, QA_TESTS_PROMPT
from observability.logs import error_class, log_event
from orchestration.a2a import ReviewRequest
from orchestration.handoff import qa_test_file
from orchestration.project_state import Task, TaskHandoff
from orchestration.review import Issue, QAReport, QATestSuite, TestCorrection, TestOutcome
from tools.exec_python import ExecOutcome, exec_python
from tools.file_tools import write_file
from tools.guarded import guarded_tool
from tools.junit import CaseResult, parse_junit
from tools.specs import ARG_MODELS
from tools.workspace import ToolError, Workspace, sha256_file

MAX_ASSERTION = 500
_TAIL = 1500
# A file:line token that starts a path (so absolute stdlib paths don't match mid-way)
_LOCATION = re.compile(r"(?<![\w/.\\-])([\w-][\w./-]*\.py):(\d+)")


class QAError(Exception):
    """QA could not produce a valid test suite (model error, or two invalid suites)."""


class Fix(BaseModel):
    test: str
    root_cause: str
    fix: str


class FixPlan(BaseModel):
    fixes: list[Fix]


class TestVerdict(BaseModel):
    __test__ = False

    test: str
    test_is_wrong: bool
    reason: str
    corrected: str = ""


class TestAudit(BaseModel):
    """QA's round-1 check of its own failing tests against the criteria."""

    __test__ = False

    verdicts: list[TestVerdict]


def canonical_suite_path(settings: Settings, task_id: str) -> Path:
    """QA's own copy of a suite, outside the Coder's workspace; workspace copies are restored from it."""
    return settings.state_dir / "qa" / "suites" / qa_test_file(task_id)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _clip(text: str, limit: int = MAX_ASSERTION) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _assertion(case: CaseResult) -> str:
    """The failing source line plus pytest's E lines; else the message; else the text's tail."""
    lines = case.text.splitlines()
    source = [ln[1:].strip() for ln in lines if ln.startswith(">")]
    errors = [ln[1:].strip() for ln in lines if ln.startswith("E ")]
    if errors:
        return _clip("\n".join(source[-1:] + errors))
    return _clip(case.message or case.text[-MAX_ASSERTION:] or case.outcome)


def _location(text: str, fallback: str) -> str:
    """The last file.py:line in the failure text that isn't a test_* file."""
    hits = [f"{m[1]}:{m[2]}" for m in _LOCATION.finditer(text) if not Path(m[1]).name.startswith("test_")]
    return hits[-1] if hits else fallback


def _replace_tests(content: str, replacements: dict[str, str]) -> str | None:
    """Swap whole top-level test functions by name; every other line of the module is kept as is.
    None if a name is not defined in the module or a replacement is not exactly one function of that name."""
    funcs = (ast.FunctionDef, ast.AsyncFunctionDef)
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return None
    spans = {n.name: (min([n.lineno, *(d.lineno for d in n.decorator_list)]) - 1, n.end_lineno)
             for n in tree.body if isinstance(n, funcs) and n.name in replacements}
    if set(spans) != set(replacements):
        return None
    for name, source in replacements.items():
        try:
            body = ast.parse(source).body
        except SyntaxError:
            return None
        if len(body) != 1 or not isinstance(body[0], funcs) or body[0].name != name:
            return None
    lines = content.splitlines(keepends=True)
    for name, (start, end) in sorted(spans.items(), key=lambda kv: -kv[1][0]):
        lines[start:end] = [replacements[name].strip("\n") + "\n"]
    return "".join(lines)


def _approved(r: dict[str, Any]) -> bool:
    """The QAReport validator's conditions, computed rather than asserted."""
    return (
        r["tests_run"] >= 1 and r["passed"] == r["tests_run"] and r["failed"] == r["errors"] == r["skipped"] == 0
        and not r["missing"] and r["exit_code"] == 0 and not r["timed_out"] and r["qa_error"] is None
        and not any(i.severity == "High" for i in r["issues"])
    )


def _report(**fields: Any) -> QAReport:
    return QAReport(**fields, approved=_approved(fields))


@dataclass
class _Run:
    """What reconciling one pytest run against the suite manifests found."""

    results: list[TestOutcome] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def _later_tasks(later: Sequence[Task]) -> str:
    """The suite prompt's later-tasks section, or nothing for the last task."""
    if not later:
        return ""
    lines = ["Later tasks (built after this one; do not test behaviour they define or change):"]
    for t in later:
        lines += [f"{t.task_id}: {t.title}", *(f"  - {c}" for c in t.acceptance_criteria)]
    return "\n".join(lines) + "\n\n"


class QAAgent:
    def __init__(self, settings: Settings | None = None, *, model: BaseChatModel | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.model = resilient(model, self.settings, "qa") if model is not None else None

    def _structured(self, schema: type) -> Any:
        if self.model is None:
            self.model = make_model(self.settings, "qa")
        # suite, fix plan and audit are side-effect free and run-stable: response-cache eligible
        return self.model.cached().with_structured_output(schema, include_raw=True)

    @property
    def _ws(self) -> Workspace:
        return Workspace(self.settings.workspace_dir)

    @property
    def _qa_dir(self) -> Path:
        return self.settings.state_dir / "qa"

    # --- suite generation -------------------------------------------------------------

    def generate_suite(self, handoff: TaskHandoff, files: dict[str, str], later: Sequence[Task] = ()) -> QATestSuite:
        """Ask for a suite; on an invalid one retry once with the problems appended. `later` are the
        tasks built after this one: QA sees them so its frozen suite does not pin behaviour they change."""
        try:
            structured = self._structured(QATestSuite)
        except Exception as e:  # missing provider package, bad model id
            raise QAError(f"QA model setup failed: {type(e).__name__}: {e}") from e
        criteria = "\n".join(f"{i}. {c}" for i, c in enumerate(handoff.acceptance_criteria, 1))
        code = "\n\n".join(f"--- {path} ---\n{text}" for path, text in sorted(files.items())) or "(no files)"
        messages: list[Any] = [
            SystemMessage(QA_TESTS_PROMPT),
            HumanMessage(f"Task {handoff.task_id}: {handoff.title}\n{handoff.description}\n\n"
                         f"Acceptance criteria:\n{criteria}\n\n{_later_tasks(later)}Delivered files:\n{code}"),
        ]
        problems: list[str] = []
        for attempt in range(2):
            try:
                out = structured.invoke(messages)
            except Exception as e:
                raise QAError(f"QA model call failed: {type(e).__name__}: {e}") from e
            suite = out.get("parsed")
            if cut_off(out.get("raw")):  # the parse error of a truncated reply would mislead the retry
                problems = ["your reply hit the output-token limit and was cut off before it was complete. "
                            "Write a shorter suite: fewer, compact tests (pytest.mark.parametrize for "
                            "similar cases), no long docstrings or comments"]
            elif out.get("parsing_error") is not None or suite is None:
                problems = [str(out.get("parsing_error") or "the model returned no structured suite")]
            else:
                problems = suite.check(len(handoff.acceptance_criteria))
                if not problems:
                    return suite
            if attempt == 0:
                messages.append(HumanMessage(
                    "Your test suite was invalid:\n- " + "\n- ".join(problems)
                    + "\n\nReturn a corrected suite that fixes these problems."))
        raise QAError("QA produced an invalid test suite twice: " + "; ".join(problems))

    # --- running suites ---------------------------------------------------------------

    def _restore(self, suites: dict[str, QATestSuite]) -> tuple[dict[str, str], bool]:
        """Write each suite's state-held content to its canonical file and the workspace root.
        Returns ({id: sha}, tampered): code the Coder runs could rewrite either copy on disk."""
        ws, shas, tampered = self._ws, {}, False
        for task_id, suite in suites.items():
            content = suite.content
            path = self._checked_write_args(qa_test_file(task_id), content)
            canonical = canonical_suite_path(self.settings, task_id)
            if not canonical.is_file() or canonical.read_text(encoding="utf-8", errors="replace") != content:
                tampered = True
                canonical.parent.mkdir(parents=True, exist_ok=True)
                canonical.write_text(content, encoding="utf-8")
            target = ws.resolve(path)
            if not target.is_file() or target.read_text(encoding="utf-8", errors="replace") != content:
                tampered = True
            guarded_tool("write_file", lambda: write_file(ws, path, content))
            shas[task_id] = _sha(content)
        return shas, tampered

    def _checked_write_args(self, path: str, content: str) -> str:
        """QA's direct write_file calls get the Coder's argument model plus the workspace check up front
        (spec §5.7), so a bad path never reaches write_file. Raises ToolError (or ValidationError)."""
        a = ARG_MODELS["write_file"].model_validate({"path": path, "content": content})
        self._ws.resolve(a.path)  # WorkspaceError if it escapes the workspace
        return a.path

    def _pytest(self, files: list[str], junit: Path) -> ExecOutcome:
        """One sandboxed pytest call over every suite file. exec_python's pytest mode takes a single
        target, so a script snippet calls pytest.main with all of them under the same timeout."""
        # QA's own empty ini: the Coder's pytest.ini / pyproject addopts must not reshape QA's run
        ini = self._qa_dir / "pytest.ini"
        ini.parent.mkdir(parents=True, exist_ok=True)
        ini.write_text("[pytest]\n", encoding="utf-8")
        args = ["-q", "-p", "no:cacheprovider", "-c", str(ini), f"--rootdir={self._ws.root}",
                "--continue-on-collection-errors", f"--junitxml={junit}", *files]
        code = f"import sys\nimport pytest\n\nsys.exit(int(pytest.main({json.dumps(args)})))\n"
        mode, timeout = "script", self.settings.exec_timeout_s
        if mode != "script" or not 0 < timeout <= self.settings.exec_timeout_s:
            raise ToolError(f"QA exec_python needs mode 'script' and 0 < timeout <= {self.settings.exec_timeout_s:g}")
        return guarded_tool("exec_python", lambda: exec_python(
            self._ws, record_dir=self._qa_dir / "exec", code=code, mode=mode, timeout=timeout))

    def run_suites(self, suites: dict[str, QATestSuite], *, task_id: str, round: int) -> QAReport:
        """Restore every suite from canonical, run them in one pytest call, reconcile with the manifests.
        Never calls the model; each issue's fix is the fallback text."""
        if not suites:
            raise ValueError("run_suites needs at least one suite")  # pytest would collect the whole workspace
        return self._run(suites, task_id=task_id, round=round, criteria={})

    def _run(self, suites: dict[str, QATestSuite], *, task_id: str, round: int,
             criteria: dict[str, list[str]]) -> QAReport:
        shas, tampered = self._restore(suites)
        self._qa_dir.mkdir(parents=True, exist_ok=True)
        junit = self._qa_dir / f"junit-{task_id.lower()}-r{round}.xml"
        junit.unlink(missing_ok=True)
        out = self._pytest([qa_test_file(t) for t in suites], junit)
        run = self._reconcile(suites, task_id, criteria, out, junit)
        count = {k: sum(r.outcome == k for r in run.results) for k in ("passed", "failed", "error", "skipped")}
        return _report(
            task_id=task_id, round=round, test_file=qa_test_file(task_id) if task_id in suites else "",
            suite_sha=shas.get(task_id, ""), suites_run=shas, tested_files={},
            tests_run=len(run.results), passed=count["passed"], failed=count["failed"], errors=count["error"],
            skipped=count["skipped"], missing=run.missing, exit_code=out.exit_code, timed_out=out.timed_out,
            duration_s=out.duration_s, results=run.results, issues=run.issues, tampered=tampered, qa_error=None,
        )

    def _reconcile(self, suites: dict[str, QATestSuite], task_id: str, criteria: dict[str, list[str]],
                   out: ExecOutcome, junit: Path) -> _Run:
        """Every manifest test of every suite must appear in the JUnit cases as passed (spec §6)."""
        run = _Run()
        stems = {qa_test_file(t)[:-3]: t for t in suites}
        prefix = {t: "" if t == task_id else f"regression in {t}: " for t in suites}

        def issue(owner: str | None, test: str | None, criterion: str | None, location: str, assertion: str) -> None:
            text = _clip(prefix.get(owner, "") + assertion) if owner else _clip(assertion)
            run.issues.append(Issue(severity="High", test=test, criterion=criterion, location=location,
                                    assertion=text, fix=f"Make {test or location} pass: {text}"))

        def criterion_of(owner: str, test: str) -> str | None:
            n = next((t.criterion for t in suites[owner].tests if t.name == test), None)
            texts = criteria.get(owner, [])
            if n is None:
                return None
            return texts[n - 1] if 0 < n <= len(texts) else f"{owner} criterion {n}"

        result = None if out.timed_out else parse_junit(junit)
        tail = _clip((out.stdout + "\n" + out.stderr)[-_TAIL:], MAX_ASSERTION - 80)
        if out.timed_out:
            issue(None, None, None, "QA test run", f"QA test run timed out after {self.settings.exec_timeout_s:g} s")
        elif result is None:
            issue(None, None, None, "QA test run", f"QA could not read pytest's JUnit XML (exit {out.exit_code}): {tail}")

        seen: set[tuple[str, str]] = set()
        for case in result.cases if result else []:
            owner = next((t for stem, t in stems.items()
                          if case.classname == stem or case.classname.endswith("." + stem)
                          or (not case.classname and case.name == stem)), None)
            test = case.name.split("[")[0]
            node = f"{case.classname}::{case.name}" if case.classname else case.name
            run.results.append(TestOutcome(name=node, outcome=case.outcome, message=_clip(case.message)))
            in_manifest = owner is not None and any(t.name == test for t in suites[owner].tests)
            if in_manifest:
                seen.add((owner, test))
                if case.outcome != "passed":
                    what = "skipped: " if case.outcome == "skipped" else ""
                    issue(owner, test, criterion_of(owner, test), _location(case.text, test),
                          what + _assertion(case))
            elif owner is not None and not case.classname:  # the module failed to import or collect
                issue(owner, None, None, _location(case.text, qa_test_file(owner)), _assertion(case))
            else:
                issue(owner, case.name or None, None, _location(case.text, node or "QA test run"),
                      f"unexpected test {node or '(unnamed)'} ({case.outcome}), not in any QA suite manifest: "
                      + _assertion(case))

        for owner, suite in suites.items():
            for t in suite.tests:
                if (owner, t.name) not in seen:
                    run.missing.append(t.name)  # still blocks approval
                    if result is not None:  # a timeout or unreadable XML already has its one High issue
                        issue(owner, t.name, criterion_of(owner, t.name), t.name,
                              f"{t.name} did not run (no test result for it)")

        failing = any(r.outcome in ("failed", "error") for r in run.results)
        if result is not None and out.exit_code not in (0, None) and not failing:
            issue(None, None, None, "QA test run", f"pytest exited {out.exit_code} with no failing test: {tail}")
        return run

    # --- review -----------------------------------------------------------------------

    def _fresh(self, request: ReviewRequest) -> tuple[dict[str, str], dict[str, str], list[Issue]]:
        """Compare each requested file's on-disk sha with the request. Returns (shas, contents, issues)."""
        ws, shas, contents, issues = self._ws, {}, {}, []
        for path, sha in sorted(request.files.items()):
            try:
                target = ws.resolve(path)
                disk = sha256_file(target) if target.is_file() else None
            except (ToolError, OSError):  # outside the workspace, or a name the OS rejects
                disk = None
            if disk is not None:
                shas[path] = disk
                contents[path] = target.read_text(encoding="utf-8", errors="replace")
            if disk != sha:
                now = f"sha {disk[:12]}" if disk else "missing"
                issues.append(Issue(
                    severity="High", test=None, criterion=None, location=path,
                    assertion=_clip(f"changed after review request: {path} (requested sha {sha[:12]}, on disk {now})"),
                    fix=f"Send a new review request with the current contents of {path}"))
        return shas, contents, issues

    def _fixes(self, issues: list[Issue], contents: dict[str, str]) -> list[Issue]:
        """One fix-plan call; its fixes replace the fallbacks by test name. Never raises."""
        try:
            code = "\n\n".join(f"--- {p} ---\n{t}" for p, t in sorted(contents.items())) or "(no files)"
            failures = "\n".join(f"- {i.test or i.location}: {i.assertion}" for i in issues)
            out = self._structured(FixPlan).invoke([
                SystemMessage(QA_FIX_PROMPT), HumanMessage(f"Code:\n{code}\n\nFailures:\n{failures}")])
            plan = out.get("parsed") if out.get("parsing_error") is None else None
            by_test = {f.test: f.fix for f in plan.fixes if f.fix.strip()} if plan else {}
        except Exception as e:  # keep the fallback fixes, but say so
            log_event("qa_fix_plan_fallback", level=logging.WARNING, error_class=error_class(e), message=str(e))
            return issues
        return [i.model_copy(update={"fix": by_test[i.test]}) if i.test in by_test else i for i in issues]

    def _save_suite(self, task_id: str, suite: QATestSuite) -> None:
        """The canonical copy (replacing any left by an earlier run) and its workspace copy."""
        path = self._checked_write_args(qa_test_file(task_id), suite.content)
        canonical = canonical_suite_path(self.settings, task_id)
        canonical.parent.mkdir(parents=True, exist_ok=True)
        canonical.write_text(suite.content, encoding="utf-8")
        guarded_tool("write_file", lambda: write_file(self._ws, path, suite.content))

    def _check_tests(self, handoff: TaskHandoff, suite: QATestSuite, issues: list[Issue],
                     contents: dict[str, str]) -> tuple[QATestSuite, list[TestCorrection]]:
        """Round 1 only: ask QA whether its own failing tests contradict the criteria, and splice in the
        replacements it gives. Any invalid replacement keeps the suite unchanged. Never raises."""
        manifest = {t.name for t in suite.tests}
        failing = sorted({i.test for i in issues
                          if i.test in manifest and not i.assertion.startswith("regression in ")})
        if not failing:
            return suite, []
        try:
            tree = ast.parse(suite.content)
            source = {n.name: ast.get_source_segment(suite.content, n) or "" for n in tree.body
                      if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in failing}
            failures = {i.test: i.assertion for i in issues if i.test in source}
            criteria = "\n".join(f"{n}. {c}" for n, c in enumerate(handoff.acceptance_criteria, 1))
            tests = "\n\n".join(f"--- {name} ---\n{source[name]}\nFailure:\n{failures.get(name, '')}"
                                 for name in sorted(source))
            code = "\n\n".join(f"--- {p} ---\n{t}" for p, t in sorted(contents.items())) or "(no files)"
            out = self._structured(TestAudit).invoke([
                SystemMessage(QA_CHECK_PROMPT),
                HumanMessage(f"Task {handoff.task_id}: {handoff.title}\n\nAcceptance criteria:\n{criteria}\n\n"
                             f"Failing tests:\n{tests}\n\nCode:\n{code}")])
            audit = out.get("parsed") if out.get("parsing_error") is None else None
        except Exception as e:  # keep the suite unchanged, but say so
            log_event("qa_audit_fallback", level=logging.WARNING, error_class=error_class(e), message=str(e))
            return suite, []
        wrong = {v.test: v for v in (audit.verdicts if audit else [])
                 if v.test_is_wrong and v.test in source and v.corrected.strip()}
        if not wrong:
            return suite, []
        content = _replace_tests(suite.content, {t: v.corrected for t, v in wrong.items()})
        if content is None:
            return suite, []
        corrected = suite.model_copy(update={"content": content})
        if corrected.check(len(handoff.acceptance_criteria)):
            return suite, []
        return corrected, [TestCorrection(test=t, reason=_clip(v.reason)) for t, v in sorted(wrong.items())]

    def review(self, handoff: TaskHandoff, request: ReviewRequest, round: int, suite: QATestSuite | None,
               prior: dict[str, QATestSuite], later: Sequence[Task] = ()) -> tuple[QAReport, QATestSuite | None]:
        """Spec §7 steps 1–7. Never raises for model or test failures."""
        task_id = handoff.task_id
        shas, contents, stale = self._fresh(request)
        generated = suite is None
        if suite is None:
            try:
                suite = self.generate_suite(handoff, contents, later)
            except QAError as e:
                return _report(
                    task_id=task_id, round=round, test_file=qa_test_file(task_id), suite_sha="", suites_run={},
                    tested_files=shas, tests_run=0, passed=0, failed=0, errors=0, skipped=0, missing=[],
                    exit_code=None, timed_out=False, duration_s=0.0, results=[], issues=stale, tampered=False,
                    qa_error=str(e)), None
            self._save_suite(task_id, suite)
        criteria = {task_id: list(handoff.acceptance_criteria)}
        others = {t: s for t, s in prior.items() if t != task_id}
        ran = self._run({**others, task_id: suite}, task_id=task_id, round=round, criteria=criteria)
        corrections: list[TestCorrection] = []
        if generated:
            # Before the suite freezes, QA may replace its own tests that contradict the criteria, then rerun once
            suite, corrections = self._check_tests(handoff, suite, ran.issues, contents)
            if corrections:
                self._save_suite(task_id, suite)
                ran = self._run({**others, task_id: suite}, task_id=task_id, round=round, criteria=criteria)
        issues = stale + ran.issues
        if issues:
            issues = self._fixes(issues, contents)
        fields = ran.model_dump(exclude={"approved"})
        fields.update(tested_files=shas, issues=issues, test_corrections=corrections)
        return _report(**fields), suite

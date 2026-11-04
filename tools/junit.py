"""Parse pytest's JUnit XML into per-test results."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Outcome = Literal["passed", "failed", "error", "skipped"]


@dataclass(frozen=True)
class CaseResult:
    name: str
    classname: str
    outcome: Outcome
    message: str
    text: str


@dataclass(frozen=True)
class JUnitResult:
    tests: int
    failures: int
    errors: int
    skipped: int
    cases: list[CaseResult]


def _case(el: ET.Element) -> CaseResult:
    outcome: Outcome = "passed"
    message = text = ""
    for tag, kind in (("error", "error"), ("failure", "failed"), ("skipped", "skipped")):
        child = el.find(tag)
        if child is not None:
            outcome = kind  # type: ignore[assignment]
            message = child.get("message") or ""
            text = child.text or ""
            break
    return CaseResult(el.get("name", ""), el.get("classname", ""), outcome, message, text)


def parse_junit(path: Path) -> JUnitResult | None:
    """None if the file is missing or not XML. Counts are summed over every <testsuite>."""
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return None
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    tests = failures = errors = skipped = 0
    cases: list[CaseResult] = []
    for suite in suites:
        suite_cases = [_case(el) for el in suite.iter("testcase")]
        n_err = sum(c.outcome == "error" for c in suite_cases)
        declared = int(suite.get("errors") or 0)
        for _ in range(max(declared - n_err, 0)):
            # An error the suite reports without a testcase of its own is still an error case
            suite_cases.append(CaseResult("", "", "error", "error reported by testsuite", ""))
        cases += suite_cases
        tests += max(int(suite.get("tests") or 0), len(suite_cases))
        failures += sum(c.outcome == "failed" for c in suite_cases)
        errors += sum(c.outcome == "error" for c in suite_cases)
        skipped += sum(c.outcome == "skipped" for c in suite_cases)
    return JUnitResult(tests, failures, errors, skipped, cases)

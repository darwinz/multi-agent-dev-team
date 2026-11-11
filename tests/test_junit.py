from pathlib import Path

from tools.exec_python import exec_python
from tools.file_tools import write_file
from tools.junit import parse_junit
from tools.workspace import Workspace


def run_junit(tmp_path: Path, files: dict[str, str], target: str):
    ws = Workspace(tmp_path / "ws")
    for name, body in files.items():
        write_file(ws, name, body)
    j = tmp_path / "j.xml"
    exec_python(ws, record_dir=tmp_path / "rec", timeout=30, path=target, junit_xml=j)
    return parse_junit(j)


def test_parse_counts_and_messages(tmp_path):
    res = run_junit(
        tmp_path,
        {"test_two.py": "def test_ok(): assert True\ndef test_bad(): assert 1 == 2\n"},
        "test_two.py",
    )
    assert res.tests == 2 and res.failures == 1 and res.errors == 0 and res.skipped == 0
    by_name = {c.name: c for c in res.cases}
    assert by_name["test_ok"].outcome == "passed"
    assert by_name["test_bad"].outcome == "failed"
    assert "assert 1 == 2" in by_name["test_bad"].message


def test_parse_collection_error(tmp_path):
    res = run_junit(tmp_path, {"test_imp.py": "import does_not_exist\ndef test_x(): pass\n"}, "test_imp.py")
    assert res.errors >= 1
    errs = [c for c in res.cases if c.outcome == "error"]
    assert errs and any("does_not_exist" in c.text or "does_not_exist" in c.message for c in errs)


def test_skipped_case_is_skipped(tmp_path):
    body = "import pytest\ndef test_s():\n    pytest.skip('nope')\n"
    res = run_junit(tmp_path, {"test_skip.py": body}, "test_skip.py")
    assert res.skipped == 1 and res.failures == 0
    assert [c.outcome for c in res.cases] == ["skipped"]


def test_parse_missing_or_garbage_returns_none(tmp_path):
    assert parse_junit(tmp_path / "nope.xml") is None
    bad = tmp_path / "bad.xml"
    bad.write_text("not xml <<<")
    assert parse_junit(bad) is None


def test_counts_summed_over_testsuites(tmp_path):
    p = tmp_path / "m.xml"
    p.write_text(
        '<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0">'
        '<testcase classname="a" name="t1"/></testsuite>'
        '<testsuite tests="1" failures="1" errors="0" skipped="0">'
        '<testcase classname="b" name="t2"><failure message="boom">tb</failure></testcase></testsuite></testsuites>'
    )
    res = parse_junit(p)
    assert res.tests == 2 and res.failures == 1 and len(res.cases) == 2

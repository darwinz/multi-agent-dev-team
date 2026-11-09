"""Tests for the Postman-style API check runner (app.api_check)."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from app.api_check import DEFAULT_COLLECTION, check_api, find_app, main

FIXTURES = Path(__file__).parent / "fixtures"


def test_reference_api_passes():
    results = check_api("reference_api:app", FIXTURES)
    assert len(results) == 5, results
    assert all(r.passed for r in results), results


def test_broken_api_fails_echo():
    results = {r.name: r for r in check_api("broken_api:app", FIXTURES)}
    assert results["health"].passed
    assert not results["echo non-object"].passed
    assert not results["echo no body"].passed


def test_import_error_reports_startup(tmp_path):
    (tmp_path / "bad_app.py").write_text('raise ImportError("nope")\n')
    t0 = time.monotonic()
    results = check_api("bad_app:app", tmp_path)
    assert time.monotonic() - t0 < 15
    assert len(results) == 1
    assert results[0].name == "startup" and not results[0].passed
    assert "nope" in results[0].detail


def test_never_binds_times_out_and_reaps(tmp_path, monkeypatch):
    (tmp_path / "hang_app.py").write_text(
        'import time\nprint("hello-from-hang", flush=True)\ntime.sleep(60)\n'
    )
    pids: list[int] = []
    real_popen = subprocess.Popen

    def recording_popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        pids.append(proc.pid)
        return proc

    monkeypatch.setattr(subprocess, "Popen", recording_popen)
    t0 = time.monotonic()
    results = check_api("hang_app:app", tmp_path, startup_timeout=2)
    assert time.monotonic() - t0 < 10
    assert len(results) == 1
    assert results[0].name == "startup" and not results[0].passed
    assert "hello-from-hang" in results[0].detail
    assert "timed out" in results[0].detail
    assert len(pids) == 1
    with pytest.raises(ProcessLookupError):
        os.kill(pids[0], 0)


def test_find_app(tmp_path):
    assert find_app(FIXTURES) == "broken_api:app"  # sorted order: broken_api < reference_api
    (tmp_path / "test_x.py").write_text("from fastapi import FastAPI\napp = FastAPI()\n")
    assert find_app(tmp_path) is None
    (tmp_path / "main.py").write_text("from fastapi import FastAPI\napp = FastAPI(title='t')\n")
    assert find_app(tmp_path) == "main:app"
    (tmp_path / "a_helper.py").write_text("x = 1\n")
    assert find_app(tmp_path) == "main:app"
    empty = tmp_path / "empty"
    empty.mkdir()
    assert find_app(empty) is None


APP_SRC = "from fastapi import FastAPI\napp = FastAPI()\n"


def test_find_app_package(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "main.py").write_text(APP_SRC)
    assert find_app(tmp_path) == "pkg.main:app"


def test_find_app_skips_hidden_and_test_dirs(tmp_path):
    for d in (".coder_snippets", ".pytest_cache"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "app.py").write_text(APP_SRC)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_app.py").write_text(APP_SRC)
    (tmp_path / "tests" / "app_test.py").write_text(APP_SRC)
    assert find_app(tmp_path) is None


def test_find_app_searches_only_delivered_files(tmp_path):
    (tmp_path / "app.py").write_text(APP_SRC)  # left over from an earlier run, sorts first
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "main.py").write_text(APP_SRC)
    (tmp_path / "palindrome.py").write_text("def is_palindrome(s): return s == s[::-1]\n")
    assert find_app(tmp_path) == "app:app"
    assert find_app(tmp_path, ["pkg/main.py"]) == "pkg.main:app"
    assert find_app(tmp_path, ["missing.py", "notes.txt", "pkg/main.py"]) == "pkg.main:app"
    # A delivery with no app finds nothing, even though a leftover app sits in the workspace
    assert find_app(tmp_path, ["palindrome.py"]) is None
    assert find_app(tmp_path, []) is None


def test_find_app_reference_only(tmp_path):
    src = (FIXTURES / "reference_api.py").read_text()
    (tmp_path / "reference_api.py").write_text(src)
    (tmp_path / "test_other.py").write_text("from fastapi import FastAPI\napp = FastAPI()\n")
    assert find_app(tmp_path) == "reference_api:app"


def test_collection_is_postman_v2_1():
    data = json.loads(DEFAULT_COLLECTION.read_text())
    assert data["info"]["schema"] == "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
    assert len(data["item"]) == 5
    for item in data["item"]:
        assert "request" in item
        assert item["response"]


def test_main_exit_codes(capsys):
    assert main(["--app", "reference_api:app", "--app-dir", str(FIXTURES)]) == 0
    out = capsys.readouterr().out
    assert out.count("PASS") == 5
    assert main(["--app", "broken_api:app", "--app-dir", str(FIXTURES)]) == 1
    assert "FAIL echo non-object" in capsys.readouterr().out


def test_check_api_app_runs_without_api_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-leak")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2-should-not-leak")
    (tmp_path / "env_app.py").write_text("raise ImportError('stop early')\n")
    envs: list[dict] = []
    real_popen = subprocess.Popen

    def recording_popen(*args, **kwargs):
        envs.append(kwargs.get("env"))
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", recording_popen)
    check_api("env_app:app", tmp_path, startup_timeout=5)
    assert len(envs) == 1 and envs[0] is not None
    assert "ANTHROPIC_API_KEY" not in envs[0] and "LANGSMITH_API_KEY" not in envs[0]
    assert "PATH" in envs[0]

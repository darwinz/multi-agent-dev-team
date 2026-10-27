import os
import re
import time
from pathlib import Path

import pytest

from tools.exec_python import MAX_OUTPUT_CHARS, exec_python
from tools.file_tools import write_file
from tools.workspace import Workspace, WorkspaceError, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]

SPAWN_SNIPPET = """
import subprocess, sys
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
open("child.pid", "w").write(str(child.pid))
while True:
    pass
"""


def run(ws, tmp_path, **kw):
    return exec_python(ws, record_dir=tmp_path / "rec", timeout=kw.pop("timeout", 5), **kw)


def test_script_success_and_record(tmp_path):
    ws = Workspace(tmp_path / "ws")
    write_file(ws, "hello.py", "print('hi')")
    out = run(ws, tmp_path, path="hello.py")
    assert out.ok and out.stdout.strip() == "hi" and out.mode == "script"
    assert out.loaded == {"hello.py": sha256_file(ws.root / "hello.py")}


def test_sibling_import_recorded(tmp_path):
    ws = Workspace(tmp_path / "ws")
    write_file(ws, "palindrome.py", "def is_pal(s): return s == s[::-1]")
    write_file(ws, "main.py", "from palindrome import is_pal\nprint(is_pal('aba'))")
    out = run(ws, tmp_path, path="main.py")
    assert out.ok and out.stdout.strip() == "True"
    assert set(out.loaded) == {"main.py", "palindrome.py"}


def test_pytest_mode_two_files(tmp_path):
    ws = Workspace(tmp_path / "ws")
    write_file(ws, "palindrome.py", "def is_pal(s): return s == s[::-1]")
    write_file(ws, "test_palindrome.py", "from palindrome import is_pal\ndef test_it(): assert is_pal('aba')")
    out = run(ws, tmp_path, path="test_palindrome.py")
    assert out.mode == "pytest" and out.ok and "1 passed" in out.stdout
    assert set(out.loaded) == {"test_palindrome.py", "palindrome.py"}


def test_failing_assertion_nonzero(tmp_path):
    ws = Workspace(tmp_path / "ws")
    write_file(ws, "test_bad.py", "def test_x(): assert 1 == 2")
    out = run(ws, tmp_path, path="test_bad.py")
    assert out.exit_code not in (0, None) and not out.ok


def test_infinite_loop_killed(tmp_path):
    ws = Workspace(tmp_path / "ws")
    write_file(ws, "loop.py", "while True: pass")
    t0 = time.monotonic()
    out = run(ws, tmp_path, path="loop.py", timeout=1)
    assert out.timed_out and out.exit_code is None and time.monotonic() - t0 < 4


def test_spawned_child_killed_too(tmp_path):
    ws = Workspace(tmp_path / "ws")
    write_file(ws, "spawn.py", SPAWN_SNIPPET)
    out = run(ws, tmp_path, path="spawn.py", timeout=1)
    pid = int((ws.root / "child.pid").read_text())
    assert out.timed_out
    time.sleep(0.3)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_output_flood_truncated_and_timed_out(tmp_path):  # Review Focus 2
    ws = Workspace(tmp_path / "ws")
    write_file(ws, "flood.py", "while True: print('x' * 1000)")
    out = run(ws, tmp_path, path="flood.py", timeout=1)
    assert out.timed_out and len(out.stdout) <= MAX_OUTPUT_CHARS + 50 and "[truncated]" in out.stdout


def test_stdin_is_devnull(tmp_path):  # Review Focus 3
    ws = Workspace(tmp_path / "ws")
    write_file(ws, "ask.py", "input('name? ')")
    out = run(ws, tmp_path, path="ask.py", timeout=5)
    assert not out.timed_out and "EOFError" in out.stderr


def test_invalid_utf8_output(tmp_path):  # Review Focus 4
    ws = Workspace(tmp_path / "ws")
    write_file(ws, "b.py", "import sys; sys.stdout.buffer.write(b'\\xff ok')")
    assert "ok" in run(ws, tmp_path, path="b.py").stdout


def test_inline_code_written_to_workspace(tmp_path):
    ws = Workspace(tmp_path / "ws")
    out = run(ws, tmp_path, code="print(2 + 2)")
    assert out.ok and out.stdout.strip() == "4" and out.target.startswith(".coder_snippets/")
    assert (ws.root / out.target).exists()


def test_target_outside_rejected(tmp_path):
    with pytest.raises(WorkspaceError):
        run(Workspace(tmp_path / "ws"), tmp_path, path="../x.py")


def test_no_eval_in_tools():
    for f in (REPO_ROOT / "tools").glob("*.py"):
        assert not re.search(r"\b(eval|exec)\(", f.read_text()), f

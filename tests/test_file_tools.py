import hashlib

import pytest

from tools.file_tools import read_file, write_file
from tools.workspace import ToolError, Workspace, WorkspaceError


def test_write_then_read_roundtrip(tmp_path):
    ws = Workspace(tmp_path / "ws")
    out = write_file(ws, "pkg/m.py", "a = 1\nb = 2\n")
    assert (out.path, out.lines, out.bytes) == ("pkg/m.py", 2, 12)
    assert out.sha256 == hashlib.sha256(b"a = 1\nb = 2\n").hexdigest()
    assert read_file(ws, "pkg/m.py") == "a = 1\nb = 2\n"


def test_read_missing_is_tool_error(tmp_path):
    with pytest.raises(ToolError, match="file not found: nope.py"):
        read_file(Workspace(tmp_path / "ws"), "nope.py")


def test_read_truncates(tmp_path):
    ws = Workspace(tmp_path / "ws")
    write_file(ws, "big.txt", "x" * 50)
    assert read_file(ws, "big.txt", max_chars=10) == "x" * 10 + "\n[truncated: 40 more characters]"


def test_read_binary_does_not_crash(tmp_path):  # Review Focus 4
    ws = Workspace(tmp_path / "ws")
    (ws.root / "b.bin").write_bytes(b"\xff\xfe\x00ok")
    assert "ok" in read_file(ws, "b.bin")


def test_write_outside_rejected(tmp_path):
    with pytest.raises(WorkspaceError):
        write_file(Workspace(tmp_path / "ws"), "../escape.py", "x")

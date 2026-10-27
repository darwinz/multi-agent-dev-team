import subprocess
import sys
from pathlib import Path

import pytest

from tools.workspace import ResetGeneration, RunLock, Workspace, WorkspaceBusyError, WorkspaceError

REPO_ROOT = Path(__file__).resolve().parents[1]

HOLD_LOCK_SNIPPET = """
import sys, time
from pathlib import Path
from tools.workspace import RunLock
with RunLock(Path(sys.argv[1])):
    print("locked", flush=True)
    time.sleep(30)
"""


def test_relative_and_absolute_inside(tmp_path):
    ws = Workspace(tmp_path / "workspace")
    assert ws.resolve("a/b.py") == (tmp_path / "workspace/a/b.py").resolve()
    assert ws.resolve(str(ws.root / "c.py")) == (ws.root / "c.py").resolve()


@pytest.mark.parametrize("bad", ["../x.py", "/etc/passwd", "a/../../x.py"])
def test_escape_rejected(tmp_path, bad):
    with pytest.raises(WorkspaceError):
        Workspace(tmp_path / "workspace").resolve(bad)


def test_symlink_escape_rejected(tmp_path):
    ws = Workspace(tmp_path / "workspace")
    (ws.root / "link").symlink_to(tmp_path)
    with pytest.raises(WorkspaceError):
        ws.resolve("link/outside.py")


def test_leading_workspace_prefix_is_stripped(tmp_path):  # Review Focus 1
    ws = Workspace(tmp_path / "workspace")
    assert ws.resolve("workspace/p.py") == ws.resolve("p.py")
    assert ws.resolve("./workspace/p.py") == ws.resolve("p.py")


def test_list_files_skips_hidden_and_pycache(tmp_path):
    ws = Workspace(tmp_path / "workspace")
    for rel in ["a.py", "pkg/b.py", ".coder_snippets/s.py", "__pycache__/a.pyc"]:
        (ws.root / rel).parent.mkdir(parents=True, exist_ok=True)
        (ws.root / rel).write_text("x")
    assert ws.list_files() == ["a.py", "pkg/b.py"]


def test_run_lock_is_exclusive_across_processes(tmp_path):
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLD_LOCK_SNIPPET, str(tmp_path)],
        stdout=subprocess.PIPE, text=True, cwd=REPO_ROOT,
    )
    assert holder.stdout.readline().strip() == "locked"
    try:
        with pytest.raises(WorkspaceBusyError, match="workspace busy"):
            with RunLock(tmp_path):
                pass
    finally:
        holder.kill()
        holder.wait()
    with RunLock(tmp_path):  # free again once the holder is gone
        pass


def test_generation_bumps(tmp_path):
    g = ResetGeneration(tmp_path)
    assert g.read() == 0 and g.bump() == 1 and ResetGeneration(tmp_path).read() == 1

"""exec_python: run workspace code in a subprocess with a timeout that kills the whole tree."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Literal

from tools.workspace import ToolError, Workspace

MAX_OUTPUT_CHARS = 8000
DEFAULT_TIMEOUT_S = 10.0
_RUNNER = Path(__file__).with_name("_runner.py").resolve()
_SNIPPETS = ".coder_snippets"


@dataclass(frozen=True)
class ExecOutcome:
    target: str
    mode: Literal["script", "pytest"]
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    duration_s: float
    loaded: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def to_observation(self) -> str:
        return "\n".join([
            f"target={self.target} mode={self.mode}",
            f"exit_code={self.exit_code}",
            f"timed_out={self.timed_out}" + (" (process tree killed)" if self.timed_out else ""),
            f"duration={self.duration_s:.2f}s",
            "--- stdout ---",
            self.stdout,
            "--- stderr ---",
            self.stderr,
        ])


class _BoundedReader(threading.Thread):
    """Drains a pipe so the child never blocks, keeping only the first MAX_OUTPUT_CHARS bytes."""

    def __init__(self, stream: IO[bytes]) -> None:
        super().__init__(daemon=True)
        self._stream = stream
        self._kept = bytearray()
        self._truncated = False

    def run(self) -> None:
        while chunk := self._stream.read(65536):
            room = MAX_OUTPUT_CHARS - len(self._kept)
            if room > 0:
                self._kept += chunk[:room]
            if len(chunk) > max(room, 0):
                self._truncated = True

    def text(self) -> str:
        out = self._kept.decode("utf-8", errors="replace")
        return out + "\n[truncated]" if self._truncated else out


def _next_snippet(ws: Workspace) -> Path:
    d = ws.root / _SNIPPETS
    d.mkdir(exist_ok=True)
    n = 1
    while (d / f"snippet_{n}.py").exists():
        n += 1
    return d / f"snippet_{n}.py"


def _kill_tree(proc: subprocess.Popen) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def exec_python(
    ws: Workspace,
    *,
    record_dir: Path,
    path: str | None = None,
    code: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    mode: Literal["auto", "script", "pytest"] = "auto",
) -> ExecOutcome:
    if (path is None) == (code is None):
        raise ToolError("exec_python needs exactly one of `path` or `code`")
    if code is not None:
        target = _next_snippet(ws)
        target.write_text(code, encoding="utf-8")
    else:
        target = ws.resolve(path)
        if not target.is_file():
            raise ToolError(f"file not found: {ws.relpath(target)}")
    if mode == "auto":
        name = target.name
        is_test = name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py"))
        mode = "pytest" if is_test else "script"

    record_dir.mkdir(parents=True, exist_ok=True)
    record = record_dir / f"{uuid.uuid4().hex}.json"
    cmd = [sys.executable, "-s", str(_RUNNER), mode, str(target), str(ws.root), str(record)]
    group = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        if os.name == "nt"
        else {"start_new_session": True}
    )

    # Bytecode goes to a cache outside the workspace; the runner drops the workspace's part of it
    # before importing anything (see _runner.py), so workspace code always compiles from source
    env = {**os.environ, "PYTHONPYCACHEPREFIX": str(record_dir / "pycache")}

    t0 = time.monotonic()
    proc = subprocess.Popen(
        cmd, cwd=ws.root, stdin=subprocess.DEVNULL, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, **group,
    )
    readers = [_BoundedReader(proc.stdout), _BoundedReader(proc.stderr)]
    for r in readers:
        r.start()
    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
    duration = time.monotonic() - t0
    # The process group belongs to this run alone: anything it spawned dies with it, success or not
    _kill_tree(proc)
    proc.wait()
    for r in readers:
        # A descendant that escaped the process group could keep a pipe open; don't wait on it forever
        r.join(timeout=2)

    try:
        loaded = json.loads(record.read_text())["loaded"]
    except (OSError, ValueError, KeyError):
        loaded = {}
    record.unlink(missing_ok=True)
    return ExecOutcome(
        target=ws.relpath(target),
        mode=mode,
        exit_code=None if timed_out else proc.returncode,
        stdout=readers[0].text(),
        stderr=readers[1].text(),
        timed_out=timed_out,
        duration_s=round(duration, 3),
        loaded=loaded,
    )

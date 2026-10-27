"""Workspace confinement, the cross-process run lock, and the memory reset generation."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path, PurePosixPath

from filelock import FileLock, Timeout


class ToolError(Exception):
    """A tool call failed; the message is shown to the model as the observation."""


class WorkspaceError(ToolError):
    """A path argument or execution target lies outside the workspace."""


class WorkspaceBusyError(Exception):
    """Another Coder run (or a memory reset) holds the workspace lock."""


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class Workspace:
    """The one directory the agent's file tools and execution targets may touch."""

    def __init__(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.root = root.resolve()

    def resolve(self, path: str) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            # Models often prefix paths with the workspace dir's own name (seen in Lab 1.3)
            parts = PurePosixPath(path.replace("\\", "/")).parts
            if parts and parts[0] == "workspace":
                candidate = Path(*parts[1:]) if len(parts) > 1 else Path(".")
        resolved = (self.root / candidate).resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise WorkspaceError(f"path {path!r} is outside the workspace {self.root}")
        return resolved

    def relpath(self, p: Path) -> str:
        return p.resolve().relative_to(self.root).as_posix()

    def list_files(self) -> list[str]:
        files = []
        for p in self.root.rglob("*"):
            rel = p.relative_to(self.root)
            if any(part.startswith(".") or part == "__pycache__" for part in rel.parts):
                continue
            if p.is_file() and not p.is_symlink():
                files.append(rel.as_posix())
        return sorted(files)

    def clear(self) -> None:
        for child in self.root.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()


class RunLock:
    """One active run per workspace, across processes. Never queues: contention raises."""

    def __init__(self, state_dir: Path) -> None:
        state_dir.mkdir(parents=True, exist_ok=True)
        # Not thread-local: the web UI acquires and releases on different threadpool threads
        self._lock = FileLock(state_dir / "workspace.lock", thread_local=False)

    def __enter__(self) -> RunLock:
        try:
            self._lock.acquire(blocking=False)
        except Timeout:
            raise WorkspaceBusyError("workspace busy: another Coder run is in progress") from None
        return self

    def __exit__(self, *exc: object) -> None:
        self._lock.release()

    def is_locked_elsewhere(self) -> bool:
        try:
            with self:
                return False
        except WorkspaceBusyError:
            return True


class ResetGeneration:
    """A counter bumped by every memory reset, so other processes can drop stale buffers."""

    def __init__(self, state_dir: Path) -> None:
        state_dir.mkdir(parents=True, exist_ok=True)
        self._path = state_dir / "generation"

    def read(self) -> int:
        try:
            return int(self._path.read_text().strip() or 0)
        except FileNotFoundError:
            return 0

    def bump(self) -> int:
        value = self.read() + 1
        self._path.write_text(str(value))
        return value

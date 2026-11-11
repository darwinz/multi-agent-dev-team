"""Workspace confinement, the cross-process run lock, and the memory reset generation."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path, PurePosixPath
from typing import Self

from filelock import FileLock, Timeout

KEEP_FILE = ".gitkeep"


class ToolError(Exception):
    """A tool call failed; the message is shown to the model as the observation."""


class WorkspaceError(ToolError):
    """A path argument or execution target lies outside the workspace."""


class WorkspaceBusyError(Exception):
    """Another Coder run (or a memory reset) holds the workspace lock."""


def sha256_file(p: Path) -> str:
    """Hex sha256 of a file's bytes."""
    return hashlib.sha256(p.read_bytes()).hexdigest()


class Workspace:
    """The one directory the agent's file tools and execution targets may touch."""

    def __init__(self, root: Path) -> None:
        """Create the root directory if needed and remember its resolved path."""
        root.mkdir(parents=True, exist_ok=True)
        self.root = root.resolve()

    def resolve(self, path: str) -> Path:
        """Resolve a path inside the workspace (tolerating a leading 'workspace/'); WorkspaceError if it escapes."""
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
        """A path inside the workspace as a POSIX path relative to the root."""
        return p.resolve().relative_to(self.root).as_posix()

    def list_files(self) -> list[str]:
        """Sorted relative paths of regular files, skipping hidden entries, __pycache__ and symlinks."""
        files = []
        for p in self.root.rglob("*"):
            rel = p.relative_to(self.root)
            if any(part.startswith(".") or part == "__pycache__" for part in rel.parts):
                continue
            if p.is_file() and not p.is_symlink():
                files.append(rel.as_posix())
        return sorted(files)

    def clear(self) -> None:
        """Delete everything in the workspace except the .gitkeep placeholder."""
        for child in self.root.iterdir():
            if child.name == KEEP_FILE:  # the committed placeholder that keeps the bind-mount dir user-owned
                continue
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()


class RunLock:
    """One active run per workspace, across processes. Never queues: contention raises."""

    def __init__(self, state_dir: Path) -> None:
        """Prepare the lock file and the pid marker under state_dir."""
        state_dir.mkdir(parents=True, exist_ok=True)
        # Not thread-local: the web UI acquires and releases on different threadpool threads
        self._lock = FileLock(state_dir / "workspace.lock", thread_local=False)
        self._marker = state_dir / "running"

    def __enter__(self) -> Self:
        """Acquire the lock without waiting (WorkspaceBusyError if held) and write this process's pid marker."""
        try:
            self._lock.acquire(blocking=False)
        except Timeout:
            raise WorkspaceBusyError("workspace busy: another Coder run is in progress") from None
        self._marker.write_text(str(os.getpid()))
        return self

    def __exit__(self, *exc: object) -> None:
        """Remove the pid marker and release the lock."""
        self._marker.unlink(missing_ok=True)
        self._lock.release()

    @staticmethod
    def is_running(state_dir: Path) -> bool:
        """Whether some process holds the lock, read from its marker without touching the lock."""
        if os.name == "nt":  # os.kill(pid, 0) would terminate the process on Windows
            return RunLock(state_dir).is_locked_elsewhere()
        try:
            pid = int((state_dir / "running").read_text())
        except (OSError, ValueError):
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False  # a crashed holder left its marker behind
        except PermissionError:
            return True
        return True

    def is_locked_elsewhere(self) -> bool:
        """Whether another holder has the lock, probed by trying to acquire it."""
        try:
            with self:
                return False
        except WorkspaceBusyError:
            return True


class ResetGeneration:
    """A counter bumped by every memory reset, so other processes can drop stale buffers."""

    def __init__(self, state_dir: Path) -> None:
        """Remember where the generation counter file lives under state_dir."""
        state_dir.mkdir(parents=True, exist_ok=True)
        self._path = state_dir / "generation"

    def read(self) -> int:
        """The current generation, 0 if never bumped."""
        try:
            return int(self._path.read_text().strip() or 0)
        except FileNotFoundError:
            return 0

    def bump(self) -> int:
        """Increment the generation and return the new value."""
        value = self.read() + 1
        self._path.write_text(str(value))
        return value

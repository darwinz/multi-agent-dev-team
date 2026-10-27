"""The agent's file tools. Every path is confined to the workspace."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from tools.workspace import ToolError, Workspace


@dataclass(frozen=True)
class WriteOutcome:
    path: str
    sha256: str
    bytes: int
    lines: int


def write_file(ws: Workspace, path: str, content: str) -> WriteOutcome:
    target = ws.resolve(path)
    data = content.encode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return WriteOutcome(
        path=ws.relpath(target),
        sha256=hashlib.sha256(data).hexdigest(),
        bytes=len(data),
        lines=len(content.splitlines()),
    )


def read_file(ws: Workspace, path: str, max_chars: int = 20_000) -> str:
    target = ws.resolve(path)
    rel = ws.relpath(target)
    if not target.exists():
        raise ToolError(f"file not found: {rel}")
    if not target.is_file():
        raise ToolError(f"not a file: {rel}")
    text = target.read_bytes().decode("utf-8", errors="replace")
    if len(text) > max_chars:
        return text[:max_chars] + f"\n[truncated: {len(text) - max_chars} more characters]"
    return text

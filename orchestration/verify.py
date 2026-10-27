"""Delivery verification: every delivered file must have run, in its current form, successfully."""

from __future__ import annotations

from dataclasses import dataclass

from orchestration.state import ExecInfo, WriteInfo
from tools.workspace import Workspace, WorkspaceError, sha256_file


@dataclass
class Verification:
    verified: bool
    notes: list[str]


def verify_delivery(
    writes: dict[str, WriteInfo],
    executions: list[ExecInfo],
    workspace: Workspace,
    entry_point: str | None = None,
) -> Verification:
    ok_runs = [e for e in executions if e["ok"]]
    delivered = sorted(p for p in writes if p.endswith(".py"))
    if not delivered and entry_point is None:
        return Verification(bool(ok_runs), [] if ok_runs else ["nothing was executed successfully"])

    notes: list[str] = []
    for f in delivered:
        path = workspace.root / f
        if not path.is_file():
            notes.append(f"{f}: missing on disk")
            continue
        current = sha256_file(path)
        loaders = [e for e in ok_runs if f in e["loaded"]]
        if any(e["seq"] > writes[f]["seq"] and e["loaded"][f] == current for e in loaders):
            continue
        notes.append(f"{f}: changed after last successful run" if loaders else f"{f}: not executed")

    if entry_point is not None:
        try:
            entry = workspace.relpath(workspace.resolve(entry_point))
        except WorkspaceError:
            entry = entry_point
        written_at = writes.get(entry, {}).get("seq", 0)
        if not any(e["target"] == entry and e["seq"] > written_at for e in ok_runs):
            notes.append(f"{entry}: entry point was not run successfully")

    return Verification(not notes, notes)

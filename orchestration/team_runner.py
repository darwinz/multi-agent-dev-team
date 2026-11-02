"""TeamRunner: runs the team graph under one workspace lock and writes evidence files (spec §9)."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from uuid import uuid4

from agents.coder_agent import CoderAgent
from agents.config import Settings
from agents.events import TeamEvent
from agents.pm_agent import PMAgent
from orchestration.project_state import MAX_TASKS, initial_project_state, state_to_jsonable
from orchestration.team_graph import TeamDeps, build_team_graph
from tools.workspace import RunLock

ApiCheck = Callable[[Path, list[str]], tuple[bool, str]]  # (workspace, files this run wrote)


class TeamRunner:
    def __init__(self, settings: Settings, *, pm: PMAgent | None = None, coder: CoderAgent | None = None) -> None:
        self.settings = settings
        self.pm = pm or PMAgent(settings)
        self.coder = coder or CoderAgent(settings)
        self.run_dir: Path | None = None
        self.final_state: dict | None = None

    def reset_memory(self) -> None:
        self.coder.reset_memory()  # takes the RunLock itself: never call while stream() is running

    def run(self, requirement: str) -> dict:
        for _ in self.stream(requirement):
            pass
        assert self.final_state is not None
        return self.final_state

    def stream(self, requirement: str, *, api_check: ApiCheck | None = None) -> Iterator[TeamEvent]:
        s = self.settings
        self.run_dir = None
        self.final_state = None
        # Lock first: a busy workspace must leave no run directory behind
        with RunLock(s.state_dir):
            run_id = uuid4().hex[:12]
            run_dir = s.state_dir / "runs" / run_id
            run_dir.mkdir(parents=True)
            self.run_dir = run_dir
            (run_dir / "requirement.txt").write_text(requirement)
            yield TeamEvent(kind="start", data={"run_id": run_id, "run_dir": str(run_dir)})

            graph = build_team_graph(TeamDeps(pm=self.pm, coder=self.coder))
            values: dict = {}
            pending: list[str] = []  # snapshot files waiting for the step's post-update values
            for chunk in graph.stream(
                initial_project_state(requirement),
                stream_mode=["updates", "custom", "values"],
                version="v2",
                config={"recursion_limit": 4 * MAX_TASKS + 10},
            ):
                kind, data = chunk["type"], chunk["data"]
                if kind == "values":
                    values = data
                    for name in pending:
                        self._write_json(name, state_to_jsonable(values))
                    pending.clear()
                elif kind == "custom":
                    yield TeamEvent.model_validate(data)
                elif kind == "updates":
                    for node, update in data.items():
                        for event in self._on_update(node, update or {}, pending):
                            yield event

            state = state_to_jsonable(values)
            for name in pending:  # defensive: the graph always ends with a values chunk
                self._write_json(name, state)

            # Still under the lock, so the check and the evidence describe the same workspace
            check: dict | None = None
            if api_check is not None and state["status"] != "failed":
                passed, report = api_check(s.workspace_dir, sorted(state["code_output"]))
                (run_dir / "api-check.txt").write_text(report)
                check = {"passed": passed, "report": report}
            self._write_json("final-state.json", state)
            self.final_state = state
            yield TeamEvent(kind="final", data={"state": state, "api_check": check})

    def _on_update(self, node: str, update: dict, pending: list[str]) -> Iterator[TeamEvent]:
        if node == "pm_node":
            spec = update.get("spec")
            data = spec.model_dump(mode="json") if spec is not None else {"error": update.get("error")}
            pending.append("state-after-pm.json")
            yield TeamEvent(kind="spec", data=data)
        elif node == "consolidate_node":
            data = update["consolidation"].model_dump(mode="json")
            self._write_json("consolidation.json", data)
            yield TeamEvent(kind="consolidation", data=data)
        elif node == "dispatch_node":
            h = update["handoff"]
            assert self.run_dir is not None
            (self.run_dir / f"handoff-{h.task_id}.json").write_text(h.model_dump_json(indent=2))
            yield TeamEvent(kind="handoff", task_id=h.task_id, data=h.model_dump(mode="json"))
        elif node == "coder_node":
            ((task_id, record),) = update["task_status"].items()
            pending.append(f"state-after-{task_id}.json")
            yield TeamEvent(kind="task_done", task_id=task_id, data=record.model_dump(mode="json"))

    def _write_json(self, name: str, data: dict) -> None:
        assert self.run_dir is not None
        (self.run_dir / name).write_text(json.dumps(data, indent=2))

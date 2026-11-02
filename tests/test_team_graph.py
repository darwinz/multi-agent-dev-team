"""The PM → Coder team graph: PM output, handoff, per-task progress, failures and consolidation."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from agents.pm_agent import PMAgent
from orchestration.handoff import render_task
from orchestration.project_state import Task, TaskHandoff, initial_project_state
from orchestration.team_graph import TeamDeps, build_team_graph
from tests.fakes import (
    PAL,
    ScriptedChatModel,
    agent,
    ai,
    happy,
    make_spec,
    spec_response,
    structured,
    tool_call,
)

REQ = "Build a small demo API"


def run(pm_responses, coder, requirement=REQ):
    """Run the team graph to completion; return (all chunks, the PM model)."""
    pm_model = ScriptedChatModel(responses=list(pm_responses))
    graph = build_team_graph(TeamDeps(pm=PMAgent(coder.settings, model=pm_model), coder=coder))
    chunks = list(graph.stream(
        initial_project_state(requirement),
        stream_mode=["updates", "custom", "values"],
        version="v2",
        config={"recursion_limit": 100},
    ))
    return chunks, pm_model


def updates(chunks, node):
    return [c["data"][node] for c in chunks if c["type"] == "updates" and node in c["data"]]


def final(chunks):
    return [c["data"] for c in chunks if c["type"] == "values"][-1]


def extra() -> list:
    """A second-task script: writes extra.py plus a test, runs it, finishes."""
    return [
        ai("1. write extra\n2. run",
           tool_call("write_file", path="extra.py", content="X = 1\n"),
           tool_call("write_file", path="test_extra.py", content="from extra import X\n\ndef test_x():\n    assert X == 1\n"),
           tool_call("exec_python", path="test_extra.py")),
        ai("done", tool_call("finish", explanation="Adds extra", plan=["write", "run"], entry_point="test_extra.py")),
    ]


def finish_only() -> list:
    return [ai("done", tool_call("finish", explanation="nothing", plan=["finish"], entry_point="x.py"))]


def test_state_after_pm_holds_valid_task_list(settings):
    chunks, _ = run([spec_response(make_spec())], agent(settings, *happy(), *happy()))
    pm_update = updates(chunks, "pm_node")[0]
    tasks = pm_update["tasks"]
    assert tasks
    assert [Task.model_validate(t.model_dump()) for t in tasks] == tasks
    spec = pm_update["spec"]
    assert len(spec.tasks) >= 2
    for field in ("name", "description", "acceptance_criteria", "tasks"):
        assert getattr(spec, field)
    assert all(r.status == "pending" for r in pm_update["task_status"].values())


def test_handoff_precedes_coder(settings):
    chunks, _ = run([spec_response(make_spec())], agent(settings, *happy(), *happy()))
    dispatch_at = next(i for i, c in enumerate(chunks) if c["type"] == "updates" and "dispatch_node" in c["data"])
    first = chunks[dispatch_at]["data"]["dispatch_node"]
    assert first["handoff"].task_id == "T1"
    assert first["task_status"]["T1"].status == "in_progress"
    t1_custom = [i for i, c in enumerate(chunks) if c["type"] == "custom" and c["data"]["task_id"] == "T1"]
    assert t1_custom and dispatch_at < min(t1_custom)


def test_coder_consumes_handoff_fields(settings):
    coder = agent(settings, *happy(), *happy())
    run([spec_response(make_spec())], coder, requirement="Build a quokkafied demo API")
    first_call = coder.model.calls[0]
    human = "\n".join(m.content for m in first_call if isinstance(m, HumanMessage))
    t1 = make_spec().tasks[0]
    assert "T1" in human
    assert t1.description in human
    for criterion in t1.acceptance_criteria:
        assert criterion in human
    assert "high priority" in human
    assert all("quokkafied" not in str(m.content) for m in first_call)


def test_render_task_contains_fields():
    h = TaskHandoff(
        task_id="T2", title="Add echo", description="Add the /echo endpoint.",
        acceptance_criteria=["POST /echo returns the body", "Bad input gives 422"], priority="medium",
        position=2, total=3, spec_name="Demo API", spec_description="A small demo API.",
        prior_files=["app.py", "test_app.py"],
    )
    text = render_task(h)
    for needle in ("Project: Demo API", "A small demo API.", "Task T2 (2 of 3, medium priority): Add echo",
                   "Add the /echo endpoint.", "Acceptance criteria:", "- [ ] POST /echo returns the body",
                   "- [ ] Bad input gives 422", "Files from earlier tasks:", "app.py", "test_app.py"):
        assert needle in text
    assert text.index("Project:") < text.index("Task T2") < text.index("Acceptance criteria:") \
        < text.index("Files from earlier tasks:")
    assert "run all tests" in text
    empty = render_task(h.model_copy(update={"prior_files": []}))
    assert "none yet" in empty


def test_progress_after_each_task_in_order(settings):
    chunks, _ = run([spec_response(make_spec())], agent(settings, *happy(), *happy()))
    coder_updates = updates(chunks, "coder_node")
    ids = [next(iter(u["task_status"])) for u in coder_updates]
    assert ids == ["T1", "T2"]
    assert [u["task_status"][i].status for u, i in zip(coder_updates, ids)] == ["done", "done"]
    assert [u["current_task_index"] for u in coder_updates] == [1, 2]
    assert all(u["handoff"] is None for u in coder_updates)
    assert final(chunks)["status"] == "completed"


def test_code_output_accumulates(settings):
    chunks, _ = run([spec_response(make_spec())], agent(settings, *happy(), *extra()))
    end = final(chunks)
    assert {"palindrome.py", "test_palindrome.py", "extra.py", "test_extra.py"} <= set(end["code_output"])
    assert end["code_output"]["palindrome.py"] == PAL
    assert len(end["artifacts"]) == 2
    # the second task was told about the first task's files
    second = updates(chunks, "dispatch_node")[1]["handoff"]
    assert "palindrome.py" in second.prior_files


def test_coder_events_forwarded_once(settings):
    coder = agent(settings, *happy(), *happy())
    recorded: list[list] = []
    original = coder.stream

    def recording_stream(task, **kw):
        events: list = []
        recorded.append(events)
        for e in original(task, **kw):
            events.append(e)
            yield e

    coder.stream = recording_stream
    chunks, _ = run([spec_response(make_spec())], coder)
    inner_t1 = [(e.kind, e.iteration) for e in recorded[0]]
    forwarded = [c["data"] for c in chunks if c["type"] == "custom" and c["data"]["task_id"] == "T1"]
    assert all(f["kind"] == "coder" for f in forwarded)
    assert [(f["data"]["kind"], f["data"]["iteration"]) for f in forwarded] == inner_t1
    assert inner_t1[-1][0] == "result"


def test_failed_task_does_not_stop_run(settings):
    chunks, _ = run([spec_response(make_spec())], agent(settings, RuntimeError("boom"), *happy()))
    status = final(chunks)["task_status"]
    assert status["T1"].status == "failed" and status["T1"].coder_status == "error"
    assert status["T2"].status == "done"
    assert final(chunks)["status"] == "partial"


def test_pm_failure_skips_coder(settings):
    bad = make_spec().model_dump()
    for task in bad["tasks"]:
        task["covers"] = ["AC1"]
    coder = agent(settings, *happy())
    chunks, _ = run([structured("Spec", bad), structured("Spec", bad)], coder)
    end = final(chunks)
    assert end["status"] == "failed"
    assert "AC" in end["error"]
    assert not updates(chunks, "dispatch_node") and not updates(chunks, "coder_node")
    assert len(coder.model.calls) == 0


def test_more_than_8_tasks_consolidated(settings):
    s = settings.model_copy(update={"max_iterations": 2})
    scripts = [r for _ in range(8) for r in finish_only()]
    chunks, _ = run([spec_response(make_spec(11)), RuntimeError("no merge")], agent(s, *scripts))
    consolidation = updates(chunks, "consolidate_node")
    assert len(consolidation) == 1
    assert len(consolidation[0]["tasks"]) == 8
    assert consolidation[0]["consolidation"].method == "deterministic"
    assert len(updates(chunks, "coder_node")) == 8
    assert [h["handoff"].task_id for h in updates(chunks, "dispatch_node")] == [f"T{i}" for i in range(1, 9)]


def test_eight_tasks_skip_consolidation(settings):
    s = settings.model_copy(update={"max_iterations": 2})
    scripts = [r for _ in range(8) for r in finish_only()]
    chunks, _ = run([spec_response(make_spec(8))], agent(s, *scripts))
    assert not updates(chunks, "consolidate_node")
    assert len(updates(chunks, "coder_node")) == 8

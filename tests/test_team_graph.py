"""The PM → Coder ↔ QA team graph: PM output, handoff, per-task progress, failures and consolidation."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from agents.pm_agent import PMAgent
from orchestration.handoff import render_task
from orchestration.project_state import Task, TaskHandoff, initial_project_state
from orchestration.team_graph import TeamDeps, build_team_graph, team_recursion_limit
from tests.fakes import (
    PAL_REVISED,
    ScriptedChatModel,
    agent,
    ai,
    critique_response,
    make_spec,
    qa_agent,
    qa_suite_response,
    spec_response,
    structured,
    task_scripts,
    tool_call,
)
from tests.scenarios import coder_run

REQ = "Build a small demo API"


def run(pm_responses, coder, qa_responses=(), requirement=REQ):
    """Run the team graph to completion; return (all chunks, the PM model, the QA agent)."""
    pm_model = ScriptedChatModel(responses=list(pm_responses))
    qa = qa_agent(coder.settings, *qa_responses)
    graph = build_team_graph(TeamDeps(pm=PMAgent(coder.settings, model=pm_model), coder=coder, qa=qa))
    chunks = list(graph.stream(
        initial_project_state(requirement),
        stream_mode=["updates", "custom", "values"],
        version="v2",
        config={"recursion_limit": team_recursion_limit(5)},
    ))
    return chunks, pm_model, qa


def passing_tasks(n: int = 2) -> tuple[list, list]:
    """(coder script, QA script) for n tasks that each pass QA in round 1."""
    coder: list = []
    qa: list = []
    for _ in range(n):
        c, q = task_scripts()
        coder += c
        qa += q
    return coder, qa


def closing(chunks):
    """coder_node updates that closed a task."""
    return [u for u in updates(chunks, "coder_node") if u.get("coder_step") == "close"]


def updates(chunks, node):
    return [c["data"][node] for c in chunks if c["type"] == "updates" and node in c["data"]]


def final(chunks):
    return [c["data"] for c in chunks if c["type"] == "values"][-1]


def extra() -> tuple[list, list]:
    """(coder script, QA script) for a second task: writes extra.py plus a test, revises it, passes QA."""
    coder = [
        ai("1. write extra\n2. run",
           tool_call("write_file", path="extra.py", content="X = 1\n"),
           tool_call(
               "write_file",
               path="test_extra.py",
               content="from extra import X\n\ndef test_x():\n    assert X == 1\n",
           ),
           tool_call("exec_python", path="test_extra.py")),
        ai("done", tool_call("finish", explanation="Adds extra", plan=["write", "run"], entry_point="test_extra.py")),
        critique_response(),
        *coder_run({"extra.py": '"""Extra."""\nX = 1\n'}, "test_extra.py", "Documented extra"),
    ]
    qa = [qa_suite_response("from extra import X\n\n\ndef test_x_qa():\n    assert X == 1\n", (("test_x_qa", 1),))]
    return coder, qa


def finish_only() -> tuple[list, list]:
    """(coder script, QA script) for a task that delivers nothing, can't reflect and gets a QA error: fast."""
    coder = [ai("done", tool_call("finish", explanation="nothing", plan=["finish"], entry_point="x.py")),
             RuntimeError("no critique"), RuntimeError("no critique")]
    return coder, [RuntimeError("no suite")]


def finish_only_tasks(n: int) -> tuple[list, list]:
    coder: list = []
    qa: list = []
    for _ in range(n):
        c, q = finish_only()
        coder += c
        qa += q
    return coder, qa


def test_state_after_pm_holds_valid_task_list(settings):
    coder, qa = passing_tasks()
    chunks, _, _ = run([spec_response(make_spec())], agent(settings, *coder), qa)
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
    coder, qa = passing_tasks()
    chunks, _, _ = run([spec_response(make_spec())], agent(settings, *coder), qa)
    dispatch_at = next(i for i, c in enumerate(chunks) if c["type"] == "updates" and "dispatch_node" in c["data"])
    first = chunks[dispatch_at]["data"]["dispatch_node"]
    assert first["handoff"].task_id == "T1"
    assert first["task_status"]["T1"].status == "in_progress"
    t1_custom = [i for i, c in enumerate(chunks) if c["type"] == "custom" and c["data"]["task_id"] == "T1"]
    assert t1_custom and dispatch_at < min(t1_custom)


def test_coder_consumes_handoff_fields(settings):
    script, qa = passing_tasks()
    coder = agent(settings, *script)
    run([spec_response(make_spec())], coder, qa, requirement="Build a quokkafied demo API")
    first_call = coder.model.inner.calls[0]
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
    # exec_python runs one file and a subprocess-run suite is invisible to delivery verification,
    # so the handoff must say how "all tests" are run (live demo: T3 ran pytest via subprocess)
    assert "each test file by its path with exec_python" in text
    assert "subprocess" in text
    empty = render_task(h.model_copy(update={"prior_files": []}))
    assert "none yet" in empty


def test_progress_after_each_task_in_order(settings):
    coder, qa = passing_tasks()
    chunks, _, _ = run([spec_response(make_spec())], agent(settings, *coder), qa)
    coder_updates = closing(chunks)
    ids = [next(iter(u["task_status"])) for u in coder_updates]
    assert ids == ["T1", "T2"]
    assert [u["task_status"][i].status for u, i in zip(coder_updates, ids)] == ["done", "done"]
    assert [u["current_task_index"] for u in coder_updates] == [1, 2]
    assert all(u["handoff"] is None for u in coder_updates)
    assert [u["task_status"][i].qa_outcome for u, i in zip(coder_updates, ids)] == ["approved", "approved"]
    assert final(chunks)["status"] == "completed"


def test_code_output_accumulates(settings):
    c1, q1 = task_scripts()
    c2, q2 = extra()
    chunks, _, _ = run([spec_response(make_spec())], agent(settings, *c1, *c2), q1 + q2)
    end = final(chunks)
    assert {"palindrome.py", "test_palindrome.py", "extra.py", "test_extra.py"} <= set(end["code_output"])
    assert end["code_output"]["palindrome.py"] == PAL_REVISED
    # one artifact per Coder run: delivery and reflection revision for each task
    assert [a.task_id for a in end["artifacts"]] == ["T1", "T1", "T2", "T2"]
    # the second task was told about the first task's files
    second = updates(chunks, "dispatch_node")[1]["handoff"]
    assert "palindrome.py" in second.prior_files


def test_coder_events_forwarded_once(settings):
    script, qa = passing_tasks()
    coder = agent(settings, *script)
    recorded: list[list] = []
    original = coder.stream

    def recording_stream(task, **kw):
        events: list = []
        recorded.append(events)
        for e in original(task, **kw):
            events.append(e)
            yield e

    coder.stream = recording_stream
    chunks, _, _ = run([spec_response(make_spec())], coder, qa)
    assert len(recorded) == 4  # two runs per task: delivery and reflection revision
    inner_t1 = [(e.kind, e.iteration) for e in recorded[0] + recorded[1]]
    forwarded = [c["data"] for c in chunks if c["type"] == "custom" and c["data"]["task_id"] == "T1"
                 and c["data"]["kind"] == "coder"]
    assert [(f["data"]["kind"], f["data"]["iteration"]) for f in forwarded] == inner_t1
    assert inner_t1[-1][0] == "result"


def test_failed_task_does_not_stop_run(settings):
    # T1: the delivery run errors, reflection can't critique, QA errors; T2 passes
    c2, q2 = task_scripts()
    coder = [RuntimeError("boom"), RuntimeError("no critique"), RuntimeError("no critique"), *c2]
    chunks, _, _ = run([spec_response(make_spec())], agent(settings, *coder), [RuntimeError("no suite"), *q2])
    status = final(chunks)["task_status"]
    assert status["T1"].status == "failed" and status["T1"].coder_status == "error"
    assert status["T1"].qa_outcome == "no_revision"
    assert status["T2"].status == "done"
    assert final(chunks)["status"] == "partial"


def test_pm_failure_skips_coder_and_qa(settings):
    bad = make_spec().model_dump()
    for task in bad["tasks"]:
        task["covers"] = ["AC1"]
    script, qa_script = task_scripts()
    coder = agent(settings, *script)
    chunks, _, qa = run([structured("Spec", bad), structured("Spec", bad)], coder, qa_script)
    end = final(chunks)
    assert end["status"] == "failed"
    assert "AC" in end["error"]
    for node in ("dispatch_node", "coder_node", "reflect_node", "qa_node"):
        assert not updates(chunks, node)
    assert len(coder.model.inner.calls) == 0 and len(qa.model.inner.calls) == 0
    assert end["regression"] is None


def test_more_than_8_tasks_consolidated(settings):
    s = settings.model_copy(update={"max_iterations": 2})
    coder, qa = finish_only_tasks(8)
    chunks, _, _ = run([spec_response(make_spec(11)), RuntimeError("no merge")], agent(s, *coder), qa)
    consolidation = updates(chunks, "consolidate_node")
    assert len(consolidation) == 1
    assert len(consolidation[0]["tasks"]) == 8
    assert consolidation[0]["consolidation"].method == "deterministic"
    assert len(closing(chunks)) == 8
    assert [h["handoff"].task_id for h in updates(chunks, "dispatch_node")] == [f"M{i}" for i in range(1, 9)]


def test_consolidated_run_leaves_no_stale_pending_records(settings):
    s = settings.model_copy(update={"max_iterations": 2})
    coder, qa = finish_only_tasks(8)
    chunks, _, _ = run([spec_response(make_spec(11)), RuntimeError("no merge")], agent(s, *coder), qa)
    state = final(chunks)
    task_ids = {t.task_id for t in state["tasks"]}
    assert task_ids == {f"M{i}" for i in range(1, 9)}
    assert set(state["task_status"]) == task_ids
    assert all(r.status != "pending" for r in state["task_status"].values())
    assert set(state["consolidation"].merged_from) == task_ids


def test_eight_tasks_skip_consolidation(settings):
    s = settings.model_copy(update={"max_iterations": 2})
    coder, qa = finish_only_tasks(8)
    chunks, _, _ = run([spec_response(make_spec(8))], agent(s, *coder), qa)
    assert not updates(chunks, "consolidate_node")
    assert len(closing(chunks)) == 8


def test_qa_sees_later_tasks_but_coder_does_not(settings):
    script, qa_script = passing_tasks()
    coder = agent(settings, *script)
    _, _, qa = run([spec_response(make_spec())], coder, qa_script)
    t2 = make_spec().tasks[1]
    suite_prompts = ["\n".join(m.content for m in call if isinstance(m, HumanMessage)) for call in qa.model.inner.calls]
    t1_prompt, t2_prompt = suite_prompts
    assert "Task T1" in t1_prompt and t2.acceptance_criteria[0] in t1_prompt
    assert "Later tasks" not in t2_prompt
    coder_t1 = "\n".join(str(m.content) for m in coder.model.inner.calls[0])
    assert t2.acceptance_criteria[0] not in coder_t1


def test_regression_check_crash_is_recorded_not_raised(settings, caplog):
    import logging

    from resilience.errors import CircuitOpenError

    caplog.set_level(logging.WARNING, logger="mad")
    coder, qa_script = passing_tasks(1)
    qa = qa_agent(settings, *qa_script)

    def down(*a, **k):
        raise CircuitOpenError("tools circuit is open", breaker="tools")

    qa.run_suites = down
    graph = build_team_graph(TeamDeps(pm=PMAgent(settings, model=ScriptedChatModel(
        responses=[spec_response(make_spec(1))])), coder=agent(settings, *coder), qa=qa))
    end = graph.invoke(initial_project_state(REQ), config={"recursion_limit": team_recursion_limit(5)})
    assert end["status"] == "partial"
    assert end["regression"]["approved"] is False and "CircuitOpenError" in end["regression"]["qa_error"]
    t1 = end["task_status"]["T1"]
    assert t1.status == "failed" and t1.qa_outcome == "qa_error"
    assert any("regression check unavailable" in line for line in t1.unresolved)
    assert [r for r in caplog.records if r.getMessage() == "regression_check_failed"
            and r.levelno == logging.WARNING]

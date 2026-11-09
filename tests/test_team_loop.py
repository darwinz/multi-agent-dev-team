"""The per-task Coder ↔ QA loop over A2A: reflection, review rounds, limits, trust boundary, regression."""

from __future__ import annotations

import hashlib

import pytest
from langchain_core.messages import HumanMessage

from agents.config import Settings
from agents.pm_agent import PMAgent
from agents.qa_agent import QAAgent
from orchestration.a2a import Approved, Broker, FixInstruction, Message
from orchestration.handoff import make_handoff
from orchestration.project_state import (
    MAX_TASKS,
    AcceptanceCriterion,
    Spec,
    Task,
    TaskRecord,
    initial_project_state,
)
from orchestration.review import Issue, QATestSuite
from orchestration.team_graph import NOOP_RETRY, TeamDeps, build_team_graph, team_recursion_limit
from tests.fakes import (
    PAL,
    PAL_REVISED,
    TEST,
    ScriptedChatModel,
    agent,
    ai,
    critique_response,
    fix_plan_response,
    happy,
    make_spec,
    qa_agent,
    qa_suite_response,
    revise_happy,
    spec_response,
    task_scripts,
    tool_call,
)
from tests.scenarios import coder_run, feedback_scripts, nonconverging_scripts
from tools.workspace import sha256_file

REQ = "Build it"


@pytest.fixture
def settings(tmp_path) -> Settings:
    # These tests run real QA pytest through the team graph; 15 s keeps them stable under load
    return Settings.from_env(project_dir=tmp_path, exec_timeout_s=15)


def run(settings, pm, coder_resp, qa_resp, max_rounds=5, broker=None):
    """Run the team graph to completion; return (all stream chunks, the deps)."""
    deps = TeamDeps(
        pm=PMAgent(settings, model=ScriptedChatModel(responses=list(pm))),
        coder=agent(settings, *coder_resp),
        qa=qa_agent(settings, *qa_resp),
        broker=broker or Broker(),
        qa_max_iterations=max_rounds,
    )
    chunks = list(build_team_graph(deps).stream(
        initial_project_state(REQ),
        stream_mode=["updates", "custom", "values"],
        version="v2",
        config={"recursion_limit": team_recursion_limit(max_rounds)},
    ))
    return chunks, deps


def final(chunks):
    return [c["data"] for c in chunks if c["type"] == "values"][-1]


def custom(chunks, task_id=None):
    return [c["data"] for c in chunks if c["type"] == "custom" and task_id in (None, c["data"]["task_id"])]


def a2a(chunks, event=None, intent=None):
    """A2A log records, optionally filtered by event (sent/received/rejected) and intent."""
    return [e["data"] for e in custom(chunks) if e["kind"] == "a2a"
            and event in (None, e["data"]["event"]) and intent in (None, e["data"]["message"]["intent"])]


def sent(chunks, intent=None):
    return [r["message"] for r in a2a(chunks, "sent", intent)]


def label(e):
    if e["kind"] == "a2a":
        return f"a2a {e['data']['event']} {e['data']['message']['intent']}"
    return e["kind"]


def one_task():
    return [spec_response(make_spec(1))]


def bad_critique():
    out = critique_response()
    out.tool_calls[0]["args"] = {"items": []}  # violates min_length=1
    return out


def finish_only():
    return [ai("done", tool_call("finish", explanation="nothing to change", plan=["finish"]))]


def human_text(call) -> str:
    return "\n".join(str(m.content) for m in call if isinstance(m, HumanMessage))


# --- success, feedback, limits ---------------------------------------------------------

def test_success_exits_after_one_round(settings):
    coder, qa = task_scripts()
    chunks, deps = run(settings, one_task(), coder, qa)
    labels = [label(e) for e in custom(chunks, "T1")]
    loop = [x for x in labels if x != "coder"]
    assert loop == ["reflection", "a2a sent review_request", "a2a received review_request", "qa_report",
                    "a2a sent approved", "a2a received approved"]
    assert labels[0] == "coder" and labels.index("reflection") > max(i for i, x in enumerate(labels) if x == "coder")
    end = final(chunks)
    rec = end["task_status"]["T1"]
    assert rec.status == "done" and rec.qa_rounds == 1 and rec.qa_outcome == "approved" and rec.reflected
    assert rec.qa_test_file == "test_qa_t1.py"
    assert end["status"] == "completed"
    assert deps.broker.pending("coder") == deps.broker.pending("qa") == 0
    assert end["regression"]["approved"] is True


def test_reviewed_artifact_is_the_revised_one(settings):
    coder, qa = task_scripts()
    chunks, _ = run(settings, one_task(), coder, qa)
    end = final(chunks)
    (reflection,) = end["reflections"]
    assert reflection.changed == ["palindrome.py"]
    (request,) = sent(chunks, "review_request")
    assert request["payload"]["files"] == reflection.after
    assert end["qa_reports"][0].tested_files == reflection.after
    assert reflection.after["palindrome.py"] == sha256_file(settings.workspace_dir / "palindrome.py")
    assert end["code_output"]["palindrome.py"] == PAL_REVISED


def test_feedback_drives_revision(settings):
    pm, coder, qa = feedback_scripts()
    chunks, deps = run(settings, pm, coder, qa)
    end = final(chunks)
    r1, r2 = end["qa_reports"]
    assert r1.failed == 1
    assert [i.test for i in r1.issues] == ["test_reverse"]
    assert r1.test_corrections == []  # QA checked test_reverse against the criteria and kept it
    assert r1.issues[0].fix.startswith("relink each node")  # QA's fix plan, not the fallback text

    (fix,) = sent(chunks, "fix_instruction")
    assertion = FixInstruction(**fix["payload"]).issues[0].assertion
    # coder calls: deliver (2), critique (1), revision (2), then the fix run's first call
    assert assertion in human_text(deps.coder.model.inner.calls[5])

    req1, req2 = sent(chunks, "review_request")
    assert req1["payload"]["round"] == 1 and req2["payload"]["round"] == 2
    assert req1["payload"]["files"]["linked_list.py"] != req2["payload"]["files"]["linked_list.py"]
    assert r2.approved and r2.round == 2

    replies = [m for m in sent(chunks) if m["sender"] == "qa"]
    assert [m["intent"] for m in replies] == ["fix_instruction", "approved"]
    assert [m["correlation_id"] for m in replies] == [req1["correlation_id"], req2["correlation_id"]]
    rec = end["task_status"]["T1"]
    assert rec.status == "done" and rec.qa_rounds == 2 and end["status"] == "completed"


def test_max_iterations_sends_final_report(settings):
    pm, coder, qa = nonconverging_scripts(2)
    chunks, _ = run(settings, pm, coder, qa, max_rounds=2)
    end = final(chunks)
    assert len(end["qa_reports"]) == 2
    last = [m for m in sent(chunks) if m["sender"] == "qa"][-1]
    assert last["intent"] == "final_report"
    assert last["payload"]["reason"] == "max_iterations" and last["payload"]["unresolved"]
    rec = end["task_status"]["T1"]
    assert rec.status == "failed" and rec.qa_outcome == "unresolved" and rec.unresolved
    assert end["status"] == "partial"
    assert not sent(chunks, "approved")


def test_qa_error_ends_task(settings):
    coder, _ = task_scripts()
    chunks, deps = run(settings, one_task(), coder, [RuntimeError("QA model down")])
    end = final(chunks)
    (report,) = end["qa_reports"]
    assert report.qa_error
    (msg,) = sent(chunks, "final_report")
    assert msg["payload"]["reason"] == "qa_error"
    assert len(deps.coder.model.inner.calls) == 5  # deliver 2 + critique 1 + revision 2; nothing after the report
    rec = end["task_status"]["T1"]
    assert rec.status == "failed" and rec.qa_outcome == "qa_error"


def test_unexpected_qa_exception_is_qa_error(settings, monkeypatch):
    def broken_review(self, *args, **kwargs):
        raise ValueError("unreadable JUnit XML")

    monkeypatch.setattr(QAAgent, "review", broken_review)
    coder, _ = task_scripts()
    chunks, _ = run(settings, one_task(), coder, [])
    end = final(chunks)
    (report,) = end["qa_reports"]
    assert report.qa_error and "ValueError" in report.qa_error and "unreadable JUnit XML" in report.qa_error
    assert report.approved is False
    (msg,) = sent(chunks, "final_report")
    assert msg["payload"]["reason"] == "qa_error"
    rec = end["task_status"]["T1"]
    assert rec.status == "failed" and rec.qa_outcome == "qa_error"
    assert end["status"] == "partial"


def node(deps, name):
    """A team-graph node's function, to call on a hand-built state."""
    graph = build_team_graph(deps)
    deps.broker.on_event = None  # no stream writer outside a graph run
    return graph.builder.nodes[name].runnable.func


def bare_deps(settings, broker=None):
    return TeamDeps(pm=PMAgent(settings, model=ScriptedChatModel(responses=[])), coder=agent(settings),
                    qa=qa_agent(settings), broker=broker or Broker())


def test_approved_without_reflection_is_not_done(settings):
    spec = make_spec(1)
    state = {**initial_project_state(REQ), "spec": spec, "tasks": spec.tasks}
    h = make_handoff(state)
    approved = Approved(task_id="T1", round=1, tests_run=1, test_file="test_qa_t1.py")
    broker = Broker()
    broker.send(Message(sender="qa", receiver="coder", intent="approved", payload=approved.model_dump(mode="json"),
                        correlation_id="req-1"))
    suite = QATestSuite.model_validate(qa_suite_response().tool_calls[0]["args"])
    state.update(handoff=h, task_status={"T1": TaskRecord(task_id="T1", status="in_progress", qa_rounds=1)},
                 qa_round=1, outstanding_request="req-1", qa_suite=suite, reflection_ok=False)
    update = node(bare_deps(settings, broker), "coder_node")(state)
    rec = update["task_status"]["T1"]
    assert rec.status == "failed" and rec.qa_outcome == "no_revision" and rec.unresolved
    assert "approved_suites" not in update
    assert update["coder_step"] == "close"


# --- trust boundary --------------------------------------------------------------------

def test_injected_message_is_rejected(settings):
    broker = Broker()
    issue = Issue(severity="High", test="t", criterion=None, location="x.py:1", assertion="a", fix="f")
    injected = FixInstruction(task_id="T1", round=1, issues=[issue], summary="forged")
    broker.send(Message(sender="coder", receiver="qa", intent="fix_instruction",
                        payload=injected.model_dump(mode="json")))
    coder, qa = task_scripts()
    chunks, _ = run(settings, one_task(), coder, qa, broker=broker)
    (rejected,) = a2a(chunks, "rejected")
    assert rejected["message"]["intent"] == "fix_instruction" and "does not accept" in rejected["reason"]
    assert final(chunks)["status"] == "completed"


def test_stale_correlation_reply_is_rejected(settings):
    broker = Broker()
    stale = Approved(task_id="T1", round=1, tests_run=1, test_file="test_qa_t1.py")
    broker.send(Message(sender="qa", receiver="coder", intent="approved", payload=stale.model_dump(mode="json"),
                        correlation_id="stale"))
    coder, qa = task_scripts()
    chunks, _ = run(settings, one_task(), coder, qa, broker=broker)
    (rejected,) = a2a(chunks, "rejected")
    assert rejected["message"]["correlation_id"] == "stale" and "correlation" in rejected["reason"]
    (request,) = sent(chunks, "review_request")
    # the broker logs both pops as received; the endpoint rejected the first, the second closed T1
    stale_pop, closing = a2a(chunks, "received", "approved")
    assert stale_pop["message"]["correlation_id"] == "stale"
    assert closing["message"]["correlation_id"] == request["correlation_id"]
    rec = final(chunks)["task_status"]["T1"]
    assert rec.status == "done" and rec.qa_rounds == 1


def test_recursion_limit_formula():
    assert team_recursion_limit(5) == MAX_TASKS * 13 + 10
    assert team_recursion_limit(1) == MAX_TASKS * 5 + 10


# --- reflection (C3.4) -----------------------------------------------------------------

def test_critique_failure_blocks_done(settings):
    coder = [*happy(), bad_critique(), bad_critique(), bad_critique(), bad_critique()]
    chunks, deps = run(settings, one_task(), coder, [qa_suite_response()])
    end = final(chunks)
    (reflection,) = end["reflections"]
    assert reflection.error and reflection.critique is None
    (report,) = end["qa_reports"]
    # QA's tests pass, but no revision was recorded
    assert report.passed == report.tests_run >= 1, report.issues
    (msg,) = [m for m in sent(chunks) if m["sender"] == "qa"]
    assert msg["intent"] == "final_report" and msg["payload"]["reason"] == "no_revision"
    rec = end["task_status"]["T1"]
    assert rec.status == "failed" and rec.qa_outcome == "no_revision" and not rec.reflected
    assert end["status"] == "partial"


def test_noop_revision_blocks_done(settings):
    coder = [*happy(), critique_response(), *finish_only(), critique_response(), *finish_only()]
    chunks, deps = run(settings, one_task(), coder, [qa_suite_response()])
    end = final(chunks)
    (reflection,) = end["reflections"]
    assert reflection.changed == [] and reflection.error and reflection.critique is not None
    retry = human_text(deps.coder.model.inner.calls[5])  # deliver 2, critique, revision 1, critique, then the retry
    assert NOOP_RETRY not in human_text(deps.coder.model.inner.calls[3])
    assert "Your previous revision changed no files; make the changes now." in retry
    (msg,) = [m for m in sent(chunks) if m["sender"] == "qa"]
    assert msg["payload"]["reason"] == "no_revision"
    rec = end["task_status"]["T1"]
    assert rec.status == "failed" and rec.qa_outcome == "no_revision"
    assert end["status"] == "partial"


def test_noop_then_real_revision_succeeds(settings):
    coder = [*happy(), critique_response(), *finish_only(), critique_response(), *revise_happy()]
    chunks, _ = run(settings, one_task(), coder, [qa_suite_response()])
    end = final(chunks)
    (reflection,) = end["reflections"]
    assert reflection.changed == ["palindrome.py"] and reflection.error is None
    assert end["reflection_ok"] is True  # one task, so nothing has reset it
    rec = end["task_status"]["T1"]
    assert rec.status == "done" and rec.reflected and rec.qa_outcome == "approved"


# --- files sent for review -------------------------------------------------------------

def test_revision_of_one_file_keeps_all_files(settings):
    coder, qa = task_scripts()  # delivery writes both files; the revision rewrites only palindrome.py
    chunks, _ = run(settings, one_task(), coder, qa)
    end = final(chunks)
    (request,) = sent(chunks, "review_request")
    files = request["payload"]["files"]
    assert set(files) == {"palindrome.py", "test_palindrome.py"}
    delivery = next(a for a in end["artifacts"] if a.result.files.get("palindrome.py") == PAL)
    assert delivery.result.files["test_palindrome.py"] == TEST
    assert files["test_palindrome.py"] == hashlib.sha256(TEST.encode()).hexdigest()
    assert set(end["qa_reports"][0].tested_files) == {"palindrome.py", "test_palindrome.py"}
    assert set(end["task_status"]["T1"].files) == {"palindrome.py", "test_palindrome.py"}


def test_deleted_file_dropped_from_request(settings):
    revision = [
        ai("apply the self-review and drop the old test",
           tool_call("write_file", path="palindrome.py", content=PAL_REVISED),
           tool_call("exec_python", code="import os\nos.remove('test_palindrome.py')\n")),
        ai("done", tool_call("finish", explanation="revised", plan=["revise"])),
    ]
    coder = [*happy(), critique_response(), *revision]
    chunks, _ = run(settings, one_task(), coder, [qa_suite_response()])
    (request,) = sent(chunks, "review_request")
    assert list(request["payload"]["files"]) == ["palindrome.py"]
    (event,) = [e for e in custom(chunks, "T1") if e["kind"] == "reflection"]
    assert event["data"]["dropped"] == ["test_palindrome.py"]
    end = final(chunks)
    assert "test_palindrome.py" not in end["task_files"]
    assert end["task_status"]["T1"].status == "done"


# --- regressions across tasks ----------------------------------------------------------

STACK_V1 = """\
class Stack:
    def __init__(self):
        self._items = []

    def push(self, item):
        self._items.append(item)

    def pop(self):
        return self._items.pop()
"""
STACK_V1_DOC = '"""A stack."""\n\n' + STACK_V1
STACK_V2_BROKEN = STACK_V1_DOC.replace("self._items.pop()", "self._items.pop(0)") + """
    def peek(self):
        return self._items[-1]
"""
STACK_V2_BROKEN_DOC = "# peek added\n" + STACK_V2_BROKEN
STACK_V2 = STACK_V2_BROKEN_DOC.replace("self._items.pop(0)", "self._items.pop()")

CODER_STACK_TESTS = "from stack import Stack\n\n\ndef test_push():\n    s = Stack()\n    s.push(1)\n"
QA_T1 = ("from stack import Stack\n\n\ndef test_push_pop():\n    s = Stack()\n    s.push(1)\n    s.push(2)\n"
         "    assert s.pop() == 2\n    assert s.pop() == 1\n")
QA_T2 = ("from stack import Stack\n\n\ndef test_peek():\n    s = Stack()\n    s.push(1)\n    s.push(2)\n"
         "    assert s.peek() == 2\n")


def stack_spec() -> Spec:
    return Spec(
        name="Stack", description="A stack.",
        acceptance_criteria=[AcceptanceCriterion(id="AC1", text="push/pop"),
                             AcceptanceCriterion(id="AC2", text="peek")],
        tasks=[
            Task(task_id="T1", title="push and pop", description="Stack in stack.py.",
                 acceptance_criteria=["pop returns the most recently pushed item"], priority="high", covers=["AC1"]),
            Task(task_id="T2", title="peek", description="Add peek to Stack.",
                 acceptance_criteria=["peek returns the top item"], priority="high", covers=["AC2"]),
        ],
    )


def stack_coder(*t2_fixes: str) -> list:
    def run_(content, why):
        return coder_run({"stack.py": content, "test_stack.py": CODER_STACK_TESTS}, "test_stack.py", why)

    out = [*run_(STACK_V1, "stack"), critique_response(),
           *coder_run({"stack.py": STACK_V1_DOC}, "test_stack.py", "doc"),
           *run_(STACK_V2_BROKEN, "peek"), critique_response(),
           *coder_run({"stack.py": STACK_V2_BROKEN_DOC}, "test_stack.py", "doc")]
    for content in t2_fixes:
        out += coder_run({"stack.py": content}, "test_stack.py", "fix")
    return out


def _pop_fix():
    return fix_plan_response(("test_push_pop", "pop removes the first item", "pop from the end of the list"))


def test_later_task_breaking_earlier_is_caught(settings):
    qa = [qa_suite_response(QA_T1, (("test_push_pop", 1),)), qa_suite_response(QA_T2, (("test_peek", 1),)), _pop_fix()]
    chunks, _ = run(settings, [spec_response(stack_spec())], stack_coder(STACK_V2), qa)
    end = final(chunks)
    t1, t2_r1, t2_r2 = end["qa_reports"]
    assert t1.approved and not t2_r1.approved and t2_r2.approved
    assert any(i.assertion.startswith("regression in T1:") for i in t2_r1.issues)
    assert set(t2_r1.suites_run) == {"T1", "T2"}
    replies = [m["intent"] for m in sent(chunks) if m["sender"] == "qa"]
    assert replies == ["approved", "fix_instruction", "approved"]
    assert {r.status for r in end["task_status"].values()} == {"done"}
    assert end["status"] == "completed" and end["regression"]["approved"]


def test_finalize_regression_marks_failed(settings):
    qa = [qa_suite_response(QA_T1, (("test_push_pop", 1),)), qa_suite_response(QA_T2, (("test_peek", 1),)),
          _pop_fix(), _pop_fix()]
    chunks, _ = run(settings, [spec_response(stack_spec())], stack_coder(STACK_V2_BROKEN_DOC + "\n# again\n"), qa,
                    max_rounds=2)
    end = final(chunks)
    t1, t2 = end["task_status"]["T1"], end["task_status"]["T2"]
    assert t2.status == "failed" and t2.qa_outcome == "unresolved"
    assert t1.status == "failed" and t1.qa_outcome == "regressed" and t1.unresolved
    regression = end["regression"]
    assert regression is not None and not regression["approved"]
    assert set(regression["suites_run"]) == {"T1"}
    failing = [r["name"] for r in regression["results"] if r["outcome"] != "passed"]
    assert failing == ["test_qa_t1::test_push_pop"]
    assert end["status"] == "partial"


def test_no_done_task_skips_regression(settings):
    pm, coder, qa = nonconverging_scripts(1)
    chunks, _ = run(settings, pm, coder, qa, max_rounds=1)
    end = final(chunks)
    assert end["regression"] is None and end["status"] == "partial"


GOOD_STACK_TESTS = ("from stack import Stack\n\n\ndef test_push_pop():\n    s = Stack()\n    s.push(1)\n"
                    "    assert s.pop() == 1\n")
BROKEN_IMPORT_TESTS = "from stack import Stack, Missing\n\n\ndef test_peek():\n    assert Stack()\n"


def test_finalize_collection_error_regresses_only_its_task(settings):
    settings.workspace_dir.mkdir(parents=True)
    (settings.workspace_dir / "stack.py").write_text(STACK_V1)
    spec = stack_spec()
    suites = {"T1": QATestSuite(content=GOOD_STACK_TESTS, tests=[{"name": "test_push_pop", "criterion": 1}]),
              "T2": QATestSuite(content=BROKEN_IMPORT_TESTS, tests=[{"name": "test_peek", "criterion": 1}])}
    state = {**initial_project_state(REQ), "spec": spec, "tasks": spec.tasks, "approved_suites": suites,
             "task_status": {t: TaskRecord(task_id=t, status="done", qa_outcome="approved") for t in suites}}
    update = node(bare_deps(settings), "finalize")(state)
    assert set(update["task_status"]) == {"T2"}  # T1's suite still ran and passed
    t2 = update["task_status"]["T2"]
    assert t2.qa_outcome == "regressed" and any("Missing" in line for line in t2.unresolved)
    results = {r["name"]: r["outcome"] for r in update["regression"]["results"]}
    assert results.get("test_qa_t1::test_push_pop") == "passed"
    assert update["status"] == "partial"

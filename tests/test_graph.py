from langchain_core.messages import ToolMessage

from agents.result import CoderResult
from memory.semantic import SemanticMemory
from memory.window import SlidingWindow
from orchestration.graph import GraphDeps, build_graph, initial_state
from tests.fakes import PAL, TEST, FakeEmbedding, ScriptedChatModel, ai, tool_call
from tools.dispatch import ToolContext
from tools.workspace import Workspace


def deps(settings, model):
    return GraphDeps(
        model=model,
        tool_ctx=ToolContext(Workspace(settings.workspace_dir), settings.exec_timeout_s, settings.state_dir / "exec"),
        semantic=SemanticMemory(settings.chroma_dir, embedding_function=FakeEmbedding()),
        window=SlidingWindow(10),
        max_iterations=settings.max_iterations,
        top_k=settings.memory_top_k,
        max_distance=settings.memory_max_distance,
        run_id="test-run",
    )


def run_graph(d, task="write a palindrome checker"):
    events, final = [], None
    graph = build_graph(d)
    for chunk in graph.stream(initial_state(task), config={"recursion_limit": 3 * d.max_iterations + 5},
                              stream_mode=["custom", "values"], version="v2"):
        if chunk["type"] == "custom":
            events.append(chunk["data"])
        else:
            final = chunk["data"]
    return final, events


def test_happy_path_success(settings):
    m = ScriptedChatModel(responses=[
        ai("1. write module\n2. write test\n3. run test",
           tool_call("write_file", path="palindrome.py", content=PAL),
           tool_call("write_file", path="test_palindrome.py", content=TEST),
           tool_call("exec_python", path="test_palindrome.py")),  # same batch: runs the fresh files
        ai("done", tool_call("finish", explanation="Checks palindromes", plan=["write", "test"],
                             entry_point="test_palindrome.py")),
    ])
    state, events = run_graph(deps(settings, m))
    r = CoderResult.model_validate(state["result"])
    assert r.status == "success" and r.verified and r.iterations == 2
    assert r.files["palindrome.py"] == (settings.workspace_dir / "palindrome.py").read_text()
    assert r.result.exit_code == 0 and "1 passed" in r.result.stdout


def test_memory_retrieved_before_every_call(settings):
    d = deps(settings, ScriptedChatModel(responses=[
        ai("thinking"), ai("still"), ai("", tool_call("exec_python", code="print(1)")),
        ai("", tool_call("finish", explanation="e", plan=["p"])),
    ]))
    d.semantic.add("Task: earlier palindrome work", {"run_id": "old"})
    d.window.add("earlier task", "earlier answer")
    state, events = run_graph(d)
    assert [e["kind"] for e in events].count("memory") == 4 == len(d.model.calls)
    for call in d.model.calls:
        assert "earlier answer" in call[0].content and "## Recent conversation" in call[0].content


def test_iteration_limit_forced(settings):
    s = settings.model_copy(update={"max_iterations": 3})
    m = ScriptedChatModel(responses=[ai("", tool_call("exec_python", code="print(1)")) for _ in range(3)])
    state, events = run_graph(deps(s, m))
    r = CoderResult.model_validate(state["result"])
    assert r.status == "iteration_limit" and r.iterations == 3 and len(m.calls) == 3
    assert len([e for e in events if e["kind"] == "action"]) == 2  # 3rd call's tools not executed
    assert r.explanation.startswith("Stopped after 3 of 3 iterations")


def test_default_limit_is_ten(settings):
    m = ScriptedChatModel(responses=[ai("", tool_call("read_file", path="nope.py")) for _ in range(10)])
    state, _ = run_graph(deps(settings, m))
    assert state["result"]["status"] == "iteration_limit" and len(m.calls) == 10


def test_error_mid_batch_skips_rest(settings):
    m = ScriptedChatModel(responses=[
        ai("", tool_call("read_file", path="../escape"), tool_call("write_file", path="a.py", content="x")),
        ai("", tool_call("finish", explanation="e", plan=["p"])),
    ])
    state, _ = run_graph(deps(settings, m))
    obs = [msg.content for msg in state["messages"] if isinstance(msg, ToolMessage)]
    assert obs[0].startswith("error:") and obs[1] == "skipped: an earlier call in this batch failed"
    assert not (settings.workspace_dir / "a.py").exists()


def test_write_a_exec_write_b_finish_is_failed(settings):  # spec §6.4 regression via the graph
    m = ScriptedChatModel(responses=[
        ai("", tool_call("write_file", path="m.py", content="print(1)"), tool_call("exec_python", path="m.py")),
        ai("", tool_call("write_file", path="m.py", content="print(2)")),
        ai("", tool_call("finish", explanation="e", plan=["p"])),
    ])
    r = CoderResult.model_validate(run_graph(deps(settings, m))[0]["result"])
    assert r.status == "failed" and not r.verified and "m.py: changed after last successful run" in r.verification_notes


def test_unexecuted_broken_file_is_failed(settings):  # spec §6.4 regression 2 via the graph
    m = ScriptedChatModel(responses=[
        ai("", tool_call("write_file", path="palindrome.py", content="def broken(:"),
            tool_call("write_file", path="ok.py", content="print('ok')"),
            tool_call("exec_python", path="ok.py")),
        ai("", tool_call("finish", explanation="e", plan=["p"])),
    ])
    r = CoderResult.model_validate(run_graph(deps(settings, m))[0]["result"])
    assert r.status == "failed" and "palindrome.py: not executed" in r.verification_notes


def test_limit_event_separates_rejected_from_unexecuted(settings):
    s = settings.model_copy(update={"max_iterations": 1})
    bad_finish = tool_call("finish", explanation="e")  # missing plan
    state, events = run_graph(deps(s, ScriptedChatModel(responses=[ai("", bad_finish)])))
    limit = next(e for e in events if e["kind"] == "limit")
    assert limit["data"]["unexecuted"] == [] and limit["data"]["rejected"][0].startswith("finish rejected:")


def test_rejected_finish_gets_error_status_and_accurate_nudge(settings):  # review M6
    m = ScriptedChatModel(responses=[
        ai("", tool_call("finish", explanation="e")),  # missing plan
        ai("", tool_call("write_file", path="a.py", content="x"), tool_call("read_file", path="../no"),
           tool_call("write_file", path="b.py", content="y")),
        ai("", tool_call("finish", explanation="e", plan=["p"])),
    ])
    state, _ = run_graph(deps(settings, m))
    tool_msgs = [msg for msg in state["messages"] if isinstance(msg, ToolMessage)]
    assert tool_msgs[0].content.startswith("finish rejected:") and tool_msgs[0].status == "error"
    nudge = next(msg for msg in state["messages"] if msg.type == "human" and msg.content.startswith("Observation"))
    assert "no tool was called" not in nudge.content and "finish was rejected" in nudge.content
    skipped = [msg for msg in tool_msgs if msg.content.startswith("skipped")]
    assert skipped and all(msg.status == "error" for msg in skipped)


def test_py_file_generated_by_exec_is_delivered_and_verified(settings):  # review M7
    m = ScriptedChatModel(responses=[
        ai("", tool_call("exec_python", code="open('gen.py', 'w').write('print(1)\\n')")),
        ai("", tool_call("finish", explanation="e", plan=["p"])),
    ])
    r = CoderResult.model_validate(run_graph(deps(settings, m))[0]["result"])
    assert r.files == {"gen.py": "print(1)\n"} and "gen.py: not executed" in r.verification_notes

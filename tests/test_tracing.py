"""LangSmith tracing: one root per run, agent spans carrying the span metadata (spec §5.2). Fully offline."""

from __future__ import annotations

import threading
import time

import langsmith
import langsmith.run_helpers
import langsmith.run_trees
import langsmith.utils
import pytest

from agents.config import Settings
from observability.costs import CallRecord, ToolRecord
from observability import tracing
from observability.tracing import resolve_trace_url, span_metadata, tracing_enabled
from tests.scenarios import passing_scripts, scripted_runner
from tests.test_team_runner import events_of, report_of

REQ = "Build a small demo API"
_TRACING_VARS = ("LANGSMITH_TRACING", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING", "LANGCHAIN_TRACING_V2",
                 "LANGSMITH_API_KEY", "LANGCHAIN_API_KEY", "LANGSMITH_RUNS_ENDPOINTS", "LANGSMITH_OTEL_ENABLED")


@pytest.fixture
def settings(tmp_path) -> Settings:
    # These tests run real QA pytest through the team graph; 15 s keeps them stable under load
    return Settings.from_env(project_dir=tmp_path, exec_timeout_s=15)


@pytest.fixture(autouse=True)
def _clean_tracing_env(monkeypatch):
    """No ambient tracing config; langsmith caches env lookups, so clear that cache on both sides."""
    for name in _TRACING_VARS:
        monkeypatch.delenv(name, raising=False)
    langsmith.utils.get_env_var.cache_clear()
    yield
    langsmith.utils.get_env_var.cache_clear()


class NetworkAttempt(AssertionError):
    pass


class RecordingClient(langsmith.Client):
    """A LangSmith client that records runs and never touches the network.

    The tracer reaches the client through RunTree.post/patch (create_run/update_run, both overridden here),
    flush() (a no-op without a batch queue) and RunTree.get_url (info is preset; the URL is built locally).
    Every other HTTP path goes through request_with_retries or the requests session, which record and raise.
    """

    runs = None  # shadows Client.runs (the API resource, unused by the tracer) with the recorded runs

    def __init__(self) -> None:
        super().__init__(api_url="http://langsmith.invalid", api_key="test", auto_batch_tracing=False, info={})
        self.runs: dict[str, dict] = {}
        self.network: list[str] = []
        self.session.request = self._blocked  # type: ignore[method-assign]
        assert self.tracing_queue is None and self.compressed_traces is None  # nothing batches in the background

    def _blocked(self, method, url, *args, **kwargs):
        self.network.append(f"{method} {url}")
        raise NetworkAttempt(f"{method} {url}")

    def request_with_retries(self, method, pathname, *args, **kwargs):  # type: ignore[override]
        return self._blocked(method, pathname)

    def create_run(self, name, inputs, run_type, *, project_name=None, revision_id=None, **kwargs):  # type: ignore[override]
        run_id = str(kwargs["id"])
        self.runs.setdefault(run_id, {}).update(name=name, inputs=inputs, run_type=run_type,
                                                project_name=project_name, **kwargs)

    def update_run(self, run_id, **kwargs):  # type: ignore[override]
        self.runs.setdefault(str(run_id), {}).update({k: v for k, v in kwargs.items() if v is not None})

    def _construct_run_url(self, *, run, **kwargs) -> str:  # type: ignore[override]
        return f"http://langsmith.invalid/r/{run.id}"


def test_span_metadata_fields():
    def call(p: int, c: int) -> CallRecord:
        return CallRecord(ts="t", role="coder", agent="coder", model="anthropic:claude-haiku-5-5", prompt_tokens=p,
                          completion_tokens=c, cache_read_tokens=0, cache_write_tokens=0, cache_hit=False,
                          cost_usd=0.0, priced=True, duration_ms=1.0, attempts=1)

    calls = [call(100, 20), call(5, 5)]
    tools = [ToolRecord(ts="t", agent="coder", tool="write_file", ok=True, duration_ms=1, attempts=1),
             ToolRecord(ts="t", agent="coder", tool="exec_python", ok=True, duration_ms=1, attempts=1)]
    assert span_metadata("coder", calls, tools, 12.34) == {"agent_name": "coder", "token_count": 130, "tool_calls": 2,
                                                         "tool_names": ["exec_python", "write_file"], "duration_ms": 12.3}


def test_tracing_enabled_needs_flag_and_key(monkeypatch):
    assert not tracing_enabled()
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    langsmith.utils.get_env_var.cache_clear()
    assert not tracing_enabled()  # a flag without a key stays a silent no-op
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2-test")
    assert tracing_enabled()


def test_tracing_off_is_noop(settings, monkeypatch):
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False); monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    created: list[str] = []
    sent: list[str] = []

    def no_client(self, *args, **kwargs):
        created.append("Client")
        raise AssertionError("a LangSmith client was created with tracing off")

    def no_send(name):
        def _send(self, *args, **kwargs):
            sent.append(name)
            raise AssertionError(f"LangSmith {name} called with tracing off")
        return _send

    monkeypatch.setattr(langsmith.Client, "__init__", no_client)
    for name in ("create_run", "update_run", "batch_ingest_runs", "multipart_ingest", "request_with_retries"):
        monkeypatch.setattr(langsmith.Client, name, no_send(name))  # also covers a client cached by an earlier test
    runner = scripted_runner(settings, passing_scripts()); runner.run(REQ)
    assert report_of(runner)["trace_url"] is None
    assert runner.final_state["status"] == "completed"
    assert created == [] and sent == []
    ends = [e for e in events_of(runner) if e["event"] == "agent_end"]  # span metadata is logged either way
    assert {e["agent"] for e in ends} == {"pm", "coder", "qa"} and all(e["agent_name"] == e["agent"] for e in ends)


def test_one_root_and_agent_child_spans(settings, monkeypatch):
    """Capture runs with a recording LangSmith client: exactly one root, all agent spans under its trace."""
    client = RecordingClient()  # test double: records create_run/update_run kwargs, blocks every HTTP path
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2-test"); monkeypatch.setenv("LANGSMITH_TRACING", "true")
    with langsmith.run_helpers.tracing_context(enabled=True, client=client, project_name="t"):
        runner = scripted_runner(settings, passing_scripts()); runner.run(REQ)
    client.flush()
    roots = [r for r in client.runs.values() if r.get("parent_run_id") is None]
    assert len(roots) == 1 and roots[0]["name"].startswith("team-run:")
    agents = [r for r in client.runs.values() if (r.get("extra", {}).get("metadata") or {}).get("agent_name")]
    assert {r["extra"]["metadata"]["agent_name"] for r in agents} == {"pm", "coder", "qa"}
    for r in agents:
        md = r["extra"]["metadata"]
        assert {"token_count", "tool_calls", "duration_ms"} <= md.keys()
        assert r["trace_id"] == roots[0]["trace_id"]
    # The metadata sits on the LangGraph node runs themselves, not on extra child spans
    assert {r["name"] for r in agents} == {"pm_node", "coder_node", "reflect_node", "qa_node"}
    by_id = {rid: r for rid, r in client.runs.items()}
    assert all(by_id[str(r["parent_run_id"])]["name"] == "LangGraph" for r in agents)
    # Root carries run_id and nothing secret; the report and the run share the trace URL
    run_id = runner.run_dir.name
    assert roots[0]["name"] == f"team-run:{run_id}" and roots[0]["extra"]["metadata"]["run_id"] == run_id
    assert set(roots[0]["extra"]["metadata"]) <= {"run_id", "ls_method"}
    assert all(span_metadata("x", [], [], 0).keys() <= r["extra"]["metadata"].keys() for r in agents)
    assert "lsv2-test" not in repr([r.get("extra") for r in client.runs.values()])  # the key never lands in a trace
    assert report_of(runner)["trace_url"] == f"http://langsmith.invalid/r/{roots[0]['id']}"
    assert client.network == []


def traced_run(settings, client: RecordingClient, monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2-test"); monkeypatch.setenv("LANGSMITH_TRACING", "true")
    with langsmith.run_helpers.tracing_context(enabled=True, client=client, project_name="t"):
        runner = scripted_runner(settings, passing_scripts()); runner.run(REQ)
    return runner


def test_trace_url_resolved_at_run_end(settings, monkeypatch):
    client = RecordingClient()
    runner = traced_run(settings, client, monkeypatch)
    (root,) = [r for r in client.runs.values() if r.get("parent_run_id") is None]
    url = f"http://langsmith.invalid/r/{root['id']}"
    assert report_of(runner)["trace_url"] == url
    events = [e["event"] for e in events_of(runner)]
    trace_events = [e for e in events_of(runner) if e["event"] == "trace"]
    assert [e["url"] for e in trace_events] == [url]
    # Logged after the graph has run (every agent_end), not at run start
    assert events.index("trace") > max(i for i, e in enumerate(events) if e == "agent_end")
    assert client.network == []


def test_hanging_trace_url_cannot_stall_the_run(settings, monkeypatch):
    release = threading.Event()
    lookups: list[float] = []

    def hang(self):
        lookups.append(time.monotonic())
        release.wait(60)  # far past the bound; released at teardown
        return "http://langsmith.invalid/too-late"

    monkeypatch.setattr(tracing, "TRACE_URL_TIMEOUT_S", 0.05)
    monkeypatch.setattr(langsmith.run_trees.RunTree, "get_url", hang)
    try:
        client = RecordingClient()
        runner = traced_run(settings, client, monkeypatch)
        finished = time.monotonic()
        assert len(lookups) == 1 and finished - lookups[0] < 5  # the run waited the bound, not the hang
        assert report_of(runner)["trace_url"] is None
        assert runner.final_state["status"] == "completed"
        assert not [e for e in events_of(runner) if e["event"] == "trace"]
    finally:
        release.set()


def test_resolve_trace_url_never_raises():
    class Broken:
        def get_url(self):
            raise ConnectionError("blackholed")

    assert resolve_trace_url(None) is None
    assert resolve_trace_url(Broken()) is None

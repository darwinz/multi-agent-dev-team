import json

from fastapi.testclient import TestClient

from app.web import create_app
from tests.fakes import agent, ai, happy, tool_call
from tools.workspace import RunLock


def client(settings, *responses):
    return TestClient(create_app(agent(settings, *responses)), base_url="http://127.0.0.1")


def sse_events(response):
    return [json.loads(line[6:]) for line in response.iter_lines() if line.startswith("data: ")]


def test_run_streams_events_and_result(settings):
    with client(settings, *happy()).stream("POST", "/api/run", json={"task": "palindrome"}) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        events = sse_events(r)
    kinds = [e["kind"] for e in events]
    assert kinds[0] == "start" and "memory" in kinds and kinds[-1] == "result"
    assert events[-1]["data"]["result"]["status"] == "success"


def test_sequential_runs_release_the_lock(settings):
    # the stream's lock is taken in one threadpool thread and released in another
    c = client(settings, *happy(), ai("", tool_call("finish", explanation="e", plan=["p"])))
    for task in ("palindrome", "follow-up"):
        with c.stream("POST", "/api/run", json={"task": task}) as r:
            assert r.status_code == 200
            assert sse_events(r)[-1]["kind"] == "result"
    assert c.get("/api/status").json() == {"running": False}


def test_busy_is_409(settings):
    c = client(settings)
    with RunLock(settings.state_dir):
        assert c.post("/api/run", json={"task": "x"}).status_code == 409
        assert c.get("/api/status").json() == {"running": True}


def test_file_endpoints(settings):
    c = client(settings, *happy())
    c.post("/api/run", json={"task": "palindrome"})
    files = {f["path"]: f for f in c.get("/api/files").json()["files"]}
    assert files["palindrome.py"]["touched"]
    assert "def is_pal" in c.get("/api/file", params={"path": "palindrome.py"}).json()["content"]
    assert c.get("/api/file", params={"path": "../pyproject.toml"}).status_code == 400
    assert c.get("/api/file", params={"path": "nope.py"}).status_code == 404


def test_index_served(settings):
    assert "Coder" in client(settings).get("/").text


def test_foreign_host_header_rejected(settings):  # review I5: DNS rebinding
    c = client(settings)
    assert c.post("/api/run", json={"task": "x"}, headers={"host": "evil.example"}).status_code == 400
    assert c.get("/api/status", headers={"host": "localhost:8000"}).status_code == 200

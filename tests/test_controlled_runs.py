import json

from scripts.controlled_runs import main


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _a2a(path):
    """The intent of every message in the log (capability lines carry none)."""
    return [{"intent": r["message"]["intent"]} for r in _records(path) if "message" in r]


def test_script_writes_all_scenarios(tmp_path):
    out = tmp_path / "out"
    assert main(["--out", str(out)]) == 0

    fb = out / "feedback"
    r1 = json.loads((fb / "qa-T1-r1.json").read_text())
    r2 = json.loads((fb / "qa-T1-r2.json").read_text())
    assert r1["failed"] >= 1
    assert r2["approved"] is True
    log = _a2a(fb / "a2a-log.jsonl")
    assert "fix_instruction" in {m["intent"] for m in log}
    transcript = (fb / "transcript.txt").read_text()
    assert "fix_instruction" in transcript

    nc = out / "nonconverging"
    assert "final_report" in {m["intent"] for m in _a2a(nc / "a2a-log.jsonl")}
    assert "failed" in (nc / "transcript.txt").read_text()
    tb = out / "trust-boundary"
    rejected = [r for r in _records(tb / "a2a-log.jsonl") if r.get("event") == "rejected"]
    assert len(rejected) >= 2 and all(r["reason"] for r in rejected)
    assert {(r["message"]["sender"], r["message"]["intent"]) for r in rejected} == {
        ("coder", "fix_instruction"), ("qa", "approved")}
    state = json.loads((tb / "final-state.json").read_text())
    assert state["task_status"]["T1"]["status"] == "done" and state["status"] == "completed"
    assert "deliberately" in (tb / "README.md").read_text()

    for d in (fb, nc, tb):
        readme = (d / "README.md").read_text()
        assert "scripted PM/Coder/QA models, real pytest subprocess execution" in readme
        assert "exit code" in readme.lower()
        assert "Model names in transcript.txt are config defaults; no model API calls were made." in readme
        assert "claude-sonnet" not in (d / "transcript.txt").read_text()

    # No personal paths or temp roots leak into the committed evidence
    for f in out.rglob("*"):
        if f.is_file():
            text = f.read_text(errors="replace")
            assert "/Users/" not in text, f
            assert str(tmp_path) not in text, f
            assert "/private/var" not in text and "/var/folders" not in text, f


# --- v4 resilience evidence (Task 15) ---------------------------------------------------------

def _events(path):
    return _records(path)


def _named(events, name):
    return [e for e in events if e["event"] == name]


def test_resilience_runs_produce_expected_events(tmp_path):
    from scripts.resilience_runs import main as resilience_main

    out = tmp_path / "out"
    assert resilience_main(["--out", str(out)]) == 0
    assert {d.name for d in out.iterdir()} == {
        "retry-then-success", "retry-exhausted", "breaker", "cache", "invalid-output"}
    for d in out.iterdir():
        assert (d / "events.jsonl").is_file() and (d / "summary.md").is_file()

    ok = _events(out / "retry-then-success" / "events.jsonl")
    assert [e["attempt"] for e in _named(ok, "retry_attempt")] == [1]
    assert not _named(ok, "retry_exhausted")
    assert [e["attempts"] for e in _named(ok, "llm_call")] == [2]

    ex = _events(out / "retry-exhausted" / "events.jsonl")
    assert [e["attempt"] for e in _named(ex, "retry_attempt")] == [1, 2, 3]
    assert [e["attempt"] for e in _named(ex, "retry_exhausted")] == [4]
    assert all(e["level"] == "WARNING" for e in _named(ex, "retry_attempt") + _named(ex, "retry_exhausted"))
    report = json.loads((out / "retry-exhausted" / "cost-report.json").read_text())
    assert report["outcome"] == "failed" and report["calls"] == []

    bev = _events(out / "breaker" / "events.jsonl")
    br = _named(bev, "breaker_transition")
    # leg 1 (default retry_max=3): 3 calls x 4 attempts, breaker failures 1, 2, 3, closed->open, then blocked
    assert [e["attempt"] for e in _named(bev, "retry_exhausted")][:3] == [4, 4, 4]
    assert [(e["from_state"], e["to_state"]) for e in br][0] == ("closed", "open") and br[0]["failures"] == 3
    # leg 2 (retry_max=0): closed->open->half_open->closed, then closed->open->half_open->open
    assert [(e["from_state"], e["to_state"]) for e in br[1:]] == [
        ("closed", "open"), ("open", "half_open"), ("half_open", "closed"),
        ("closed", "open"), ("open", "half_open"), ("half_open", "open")]
    assert all(e["level"] == "WARNING" and e["breaker"] == "pm" for e in br)
    blocked = _named(bev, "call_blocked")
    assert blocked and blocked[0]["inner_model_calls"] == 12

    cache = [e["event"] for e in _events(out / "cache" / "events.jsonl")
             if e["event"].startswith("cache_")]
    assert cache == ["cache_miss", "cache_hit", "cache_expired"]

    inv = _events(out / "invalid-output" / "events.jsonl")
    steps = [e["step"] for e in _named(inv, "scenario_step")]
    assert steps == ["pm-probe", "qa-probe"]
    qa_part = inv[[i for i, e in enumerate(inv) if e.get("step") == "qa-probe"][0]:]
    pm_part = inv[:len(inv) - len(qa_part)]
    assert not _named(pm_part, "tool_call")
    first_qa = next(i for i, e in enumerate(qa_part) if e["event"] == "llm_call" and e.get("role") == "qa")
    assert not _named(qa_part[first_qa:], "tool_call")
    assert len(_named(inv, "run_end")) == 2
    assert "test_invalid_tool_call_rejected_before_write" in (out / "invalid-output" / "summary.md").read_text()

    for f in out.rglob("*"):
        if f.is_file():
            text = f.read_text(errors="replace")
            assert "/Users/" not in text, f
            assert str(tmp_path) not in text, f
            assert "/private/var" not in text and "/var/folders" not in text, f


def test_resilience_settings_use_scripted_models_for_every_role(tmp_path, monkeypatch):
    from scripts.resilience_runs import _settings

    for var in ("CODER_MODEL", "PM_MODEL", "QA_MODEL"):
        monkeypatch.setenv(var, "anthropic:claude-sonnet-5-5")
    settings = _settings(tmp_path)
    assert {settings.model_for(r) for r in ("pm", "coder", "qa")} == {"scripted"}

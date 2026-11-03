import json

from scripts.controlled_runs import main


def _a2a(path):
    """The intent of every message in the log (capability lines carry none)."""
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [{"intent": r["message"]["intent"]} for r in records if "message" in r]


def test_script_writes_both_scenarios(tmp_path):
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
    for d in (fb, nc):
        readme = (d / "README.md").read_text()
        assert "scripted PM/Coder/QA models, real pytest subprocess execution" in readme
        assert "exit code" in readme.lower()

    # No personal paths or temp roots leak into the committed evidence
    for f in out.rglob("*"):
        if f.is_file():
            text = f.read_text(errors="replace")
            assert "/Users/" not in text, f
            assert str(tmp_path) not in text, f
            assert "/private/var" not in text and "/var/folders" not in text, f

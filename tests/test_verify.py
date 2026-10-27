import hashlib

from orchestration.state import ExecInfo, WriteInfo
from orchestration.verify import Verification, verify_delivery
from tools.workspace import Workspace


def w(ws, path, src, seq):
    (ws.root / path).write_text(src)
    return {path: WriteInfo(seq=seq, sha256=hashlib.sha256(src.encode()).hexdigest())}


def ex(seq, target, ok, loaded):
    return ExecInfo(seq=seq, target=target, ok=ok, exit_code=0 if ok else 1, timed_out=False,
                    stdout="", stderr="", duration_s=0.1, loaded=loaded)


def test_write_a_run_write_b_is_unverified(tmp_path):  # spec §6.4 regression 1
    ws = Workspace(tmp_path / "ws")
    a = w(ws, "m.py", "print(1)", 1)
    run = ex(2, "m.py", True, {"m.py": a["m.py"]["sha256"]})
    b = w(ws, "m.py", "print(2)", 3)
    v = verify_delivery(b, [run], ws)
    assert not v.verified and v.notes == ["m.py: changed after last successful run"]


def test_unexecuted_broken_file_is_unverified(tmp_path):  # regression 2
    ws = Workspace(tmp_path / "ws")
    writes = w(ws, "palindrome.py", "def broken(:", 1) | w(ws, "ok.py", "print('ok')", 2)
    runs = [ex(3, "ok.py", True, {"ok.py": writes["ok.py"]["sha256"]})]
    v = verify_delivery(writes, runs, ws)
    assert not v.verified and "palindrome.py: not executed" in v.notes


def test_test_run_verifies_both_files(tmp_path):  # regression 3
    ws = Workspace(tmp_path / "ws")
    writes = w(ws, "palindrome.py", "def p(s): return s==s[::-1]", 1) | w(ws, "test_palindrome.py", "...", 2)
    runs = [ex(3, "test_palindrome.py", True, {k: v["sha256"] for k, v in writes.items()})]
    assert verify_delivery(writes, runs, ws, entry_point="test_palindrome.py") == Verification(True, [])


def test_failed_run_does_not_verify(tmp_path):
    ws = Workspace(tmp_path / "ws")
    writes = w(ws, "m.py", "x", 1)
    v = verify_delivery(writes, [ex(2, "m.py", False, {"m.py": writes["m.py"]["sha256"]})], ws)
    assert v.notes == ["m.py: not executed"]


def test_entry_point_must_be_run_target(tmp_path):
    ws = Workspace(tmp_path / "ws")
    writes = w(ws, "p.py", "x=1", 1) | w(ws, "main.py", "import p", 2)
    runs = [ex(3, "p.py", True, {"p.py": writes["p.py"]["sha256"]})]
    v = verify_delivery(writes, runs, ws, entry_point="main.py")
    assert "main.py: entry point was not run successfully" in v.notes

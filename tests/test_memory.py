from memory.context import format_memory_context
import pytest

from memory.semantic import MemoryHit, MemoryUnavailableError, SemanticMemory, memory_id
from memory.window import SlidingWindow, Turn
from tests.fakes import FakeEmbedding


def test_window_evicts_oldest():
    w = SlidingWindow(max_turns=2)
    for i in range(3):
        w.add(f"t{i}", f"a{i}")
    assert [t.task for t in w.turns()] == ["t1", "t2"] and len(w) == 2


def test_semantic_roundtrip_and_threshold(tmp_path):
    m = SemanticMemory(tmp_path / ".chroma", embedding_function=FakeEmbedding())
    assert m.search("anything", k=3, max_distance=0.65) == []  # empty first run is fine
    m.add("Task: palindrome checker in palindrome.py", {"run_id": "r1", "files": "palindrome.py"})
    m.add("Task: fizzbuzz printer", {"run_id": "r2", "files": "fizz.py"})
    hits = m.search("make the palindrome checker ignore punctuation", k=3, max_distance=0.65)
    assert hits and "palindrome" in hits[0].text
    assert all("fizzbuzz" not in h.text for h in hits)


def test_semantic_persists_across_instances(tmp_path):
    SemanticMemory(tmp_path / ".chroma", embedding_function=FakeEmbedding()).add("Task: x", {"run_id": "r"})
    assert SemanticMemory(tmp_path / ".chroma", embedding_function=FakeEmbedding()).count() == 1


def test_clear(tmp_path):
    m = SemanticMemory(tmp_path / ".chroma", embedding_function=FakeEmbedding())
    m.add("t", {"run_id": "r"})
    m.clear()
    assert m.count() == 0


def test_context_formatting():
    ctx = format_memory_context([Turn("t1", "a1")], [MemoryHit(id="m1", text="Task: p", distance=0.2)])
    assert "## Recent conversation" in ctx and "t1" in ctx and "Task: p" in ctx
    assert format_memory_context([], []).strip() != ""


def test_add_is_idempotent_when_response_lost_after_commit(tmp_path):
    mem = SemanticMemory(tmp_path / "c", embedding_function=FakeEmbedding())
    real = mem._collection
    state = {"n": 0}

    class LossyCollection:
        def __getattr__(self, a):
            return getattr(real, a)

        def upsert(self, **kw):
            real.upsert(**kw)
            state["n"] += 1
            if state["n"] == 1:
                raise ConnectionError("response lost")

    mem._collection = LossyCollection()
    doc_id = mem.add("Task: x", {"run_id": "r1"})
    assert state["n"] == 2 and mem.count() == 1 and doc_id == memory_id("Task: x", {"run_id": "r1"})


def test_add_with_same_content_is_one_entry(tmp_path):
    mem = SemanticMemory(tmp_path / "c", embedding_function=FakeEmbedding())
    assert mem.add("Task: x", {"run_id": "r1"}) == mem.add("Task: x", {"run_id": "r1"})
    assert mem.add("Task: x", {"run_id": "r2"}, doc_id="given") == "given"
    assert mem.count() == 2


def test_chroma_unreachable_raises_memory_unavailable(tmp_path):
    with pytest.raises(MemoryUnavailableError, match="Chroma server unreachable at 127.0.0.1:9") as raised:
        SemanticMemory(tmp_path / "c", host="127.0.0.1", port=9)
    assert raised.value.attempts == 4  # the refused connect is transient: original attempt + 3 retries

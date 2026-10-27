from memory.context import format_memory_context
from memory.semantic import MemoryHit, SemanticMemory
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

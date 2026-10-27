"""Format both memory layers into the context block injected before every model call."""

from __future__ import annotations

from memory.semantic import MemoryHit
from memory.window import Turn


def format_memory_context(turns: list[Turn], hits: list[MemoryHit]) -> str:
    if not turns and not hits:
        return "# Memory\nNo prior context found (no earlier conversation turns and no relevant past runs)."
    parts = ["# Memory"]
    if turns:
        parts.append("## Recent conversation")
        parts += [f"- User task: {t.task}\n  Your answer: {t.answer}" for t in turns]
    if hits:
        parts.append("## Relevant memories from past runs")
        parts += [f"- (distance {h.distance:.2f}) {h.text}" for h in hits]
    return "\n".join(parts)

"""Controlled A/B measurement of the Coder's conversation-history prompt cache (live, about $0.03).

    op run --env-file=.env.op -- uv run python -m scripts.cache_ab docs/evidence/v5/cache-ab.json

Both arms send the same scripted ReAct conversation through the Coder's real model wrapper: the
production system prompt and tools, a fixed memory block, and TURNS turns that each append one
tool call and a tool result of realistic size. The model's own replies are not fed back, so the
input is identical in both arms and any token difference comes from the history breakpoint alone.
Arm "before" is the Module 4 behaviour (tools and system cached); arm "after" also caches history.
Each arm starts from a fresh nonce, so neither can read the other's history entries.
"""

import json
import sys
import uuid
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agents.config import Settings
from agents.model import make_model
from agents.prompts import SYSTEM_PROMPT
from observability.costs import PRICING, usage_from_message
from tools.specs import TOOL_SPECS

TURNS = 10
MEMORY = "# Memory\n## Recent conversation\n" + "\n".join(
    f"- User task: earlier task {i}\n  Your answer: implemented and tested module_{i}.py" for i in range(40))
OUTPUT = "\n".join(f"test_bst.py::test_case_{i} PASSED" for i in range(120))  # ~1.5K tokens per turn


def run_arm(cache_history: bool) -> list[tuple[int, int, int, int]]:
    """Send the scripted conversation; return (prompt, completion, cache_read, cache_write) per call."""
    model = make_model(Settings.from_env(), "coder").bind_tools(TOOL_SPECS)
    if cache_history:
        model = model.caching_history()
    system = SystemMessage(SYSTEM_PROMPT.format(max_iterations=TURNS) + "\n\n" + MEMORY)
    history: list = [HumanMessage(f"[run {uuid.uuid4().hex}] Build bst.py with insert, search and inorder.")]
    usage = []
    for turn in range(TURNS):
        usage.append(usage_from_message(model.invoke([system, *history])))
        call_id = f"toolu_ab{turn}{uuid.uuid4().hex[:10]}"
        history += [
            AIMessage(content=f"Turn {turn}: running the tests.", tool_calls=[
                {"name": "exec_python", "args": {"path": "test_bst.py"}, "id": call_id, "type": "tool_call"}]),
            ToolMessage(f"exit_code: 0\n{OUTPUT}", tool_call_id=call_id),
        ]
    return usage


def input_cost(rows: list[tuple[int, int, int, int]], model: str) -> float:
    """USD for the prompt side only (output differs between arms only by the model's free text)."""
    price = PRICING[model]
    return sum((p - r - w) * price.input + r * price.cache_read + w * price.cache_write
               for p, _c, r, w in rows) / 1e6


def main() -> int:
    model = Settings.from_env().model_for("coder").split(":", 1)[1]
    result = {}
    for arm, on in (("before", False), ("after", True)):
        rows = run_arm(on)
        result[arm] = {
            "prompt_tokens": sum(r[0] for r in rows),
            "cache_read_tokens": sum(r[2] for r in rows),
            "cache_write_tokens": sum(r[3] for r in rows),
            "full_price_tokens": sum(r[0] - r[2] - r[3] for r in rows),
            "input_cost_usd": round(input_cost(rows, model), 6),
            "per_call": [dict(zip(("prompt", "completion", "cache_read", "cache_write"), r, strict=True))
                         for r in rows],
        }
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("cache-ab.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": model, "turns": TURNS, **result}, indent=2) + "\n", encoding="utf-8")
    for arm in ("before", "after"):
        r = result[arm]
        print(f"{arm:6}: prompt={r['prompt_tokens']:,} read={r['cache_read_tokens']:,} "
              f"write={r['cache_write_tokens']:,} full-price={r['full_price_tokens']:,} "
              f"input=${r['input_cost_usd']:.4f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

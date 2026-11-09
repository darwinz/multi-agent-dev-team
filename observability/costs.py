"""Per-call cost records, the model price table and the reconciling cost report."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

from langchain_core.messages import AIMessage
from pydantic import BaseModel

from observability.logs import log_event


class Price(BaseModel):
    """USD per 1M tokens."""

    input: float
    output: float
    cache_read: float
    cache_write: float


# Source: Anthropic first-party pricing via the claude-api skill model tables, read 2026-11-08.
# Opus/Sonnet cache reads are published ($0.20/MTok); Haiku's is derived as 0.1x input.
# cache_write is the 5-minute write rate, 1.25x input. Haiku's >100K-token prompt tier
# ($0.50 in / $2.50 out) is not modelled: the team's prompts stay well under 100K.
PRICING_AS_OF = "2026-11-08"
PRICING: dict[str, Price] = {
    "claude-opus-5-5": Price(input=4.00, output=20.00, cache_read=0.20, cache_write=5.00),
    "claude-sonnet-5-5": Price(input=2.00, output=10.00, cache_read=0.20, cache_write=2.50),
    "claude-haiku-5-5": Price(input=0.10, output=0.50, cache_read=0.01, cache_write=0.125),
}

_AGENTS = ("pm", "coder", "qa")


class CallRecord(BaseModel):
    ts: str
    role: str
    agent: str | None
    model: str
    prompt_tokens: int
    completion_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cache_hit: bool
    cost_usd: float
    priced: bool
    duration_ms: float
    attempts: int


class ToolRecord(BaseModel):
    ts: str
    agent: str | None
    tool: str
    ok: bool
    duration_ms: float
    attempts: int


def usage_from_message(msg: AIMessage) -> tuple[int, int, int, int]:
    """(prompt, completion, cache_read, cache_write); zeros when usage is absent."""
    usage = getattr(msg, "usage_metadata", None) or {}
    details = usage.get("input_token_details") or {}
    return (int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0),
            int(details.get("cache_read") or 0), int(details.get("cache_creation") or 0))


def _model_key(model: str) -> str:
    return model.split(":", 1)[1] if ":" in model else model


def cost_usd(model: str, prompt: int, completion: int, cache_read: int, cache_write: int) -> tuple[float, bool]:
    price = PRICING.get(_model_key(model))
    if price is None:
        log_event("unpriced_model", level=logging.WARNING, model=model)
        return 0.0, False
    # LangChain's input_tokens already includes the cache tokens.
    fresh = prompt - cache_read - cache_write
    total = (fresh * price.input + cache_read * price.cache_read
             + cache_write * price.cache_write + completion * price.output) / 1e6
    return total, True


def _summary(calls: list[CallRecord]) -> dict:
    return {
        "calls": len(calls),
        "prompt_tokens": sum(c.prompt_tokens for c in calls),
        "completion_tokens": sum(c.completion_tokens for c in calls),
        "cache_read_tokens": sum(c.cache_read_tokens for c in calls),
        "cache_write_tokens": sum(c.cache_write_tokens for c in calls),
        "cache_hits": sum(1 for c in calls if c.cache_hit),
        "cost_usd": round(sum(c.cost_usd for c in calls), 6),
    }


def build_cost_report(*, run_id: str, started_at: str, finished_at: str, requirement: str,
                      outcome: Literal["success", "unresolved", "failed", "aborted"],
                      models: dict[str, str], calls: list[CallRecord], trace_url: str | None,
                      error: dict | None = None) -> dict:
    pricing: dict = {"as_of": PRICING_AS_OF}
    unpriced: set[str] = set()
    for c in calls:
        key = _model_key(c.model)
        if key in PRICING:
            pricing[key] = PRICING[key].model_dump()
        else:
            unpriced.add(c.model)
    return {
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "requirement": requirement,
        "outcome": outcome,
        "error": error,
        "models": models,
        "pricing": pricing,
        "trace_url": trace_url,
        "calls": [c.model_dump() for c in calls],
        "agents": {a: _summary([c for c in calls if c.role == a]) for a in _AGENTS},
        "total": _summary(calls),
        "unpriced_models": sorted(unpriced),
    }


def write_cost_report(report: dict, reports_dir: Path, run_dir: Path) -> Path:
    text = json.dumps(report, indent=2)
    first = reports_dir / f"{report['run_id']}.json"
    for path in (first, run_dir / "cost-report.json"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return first

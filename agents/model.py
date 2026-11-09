"""Chat model construction, the resilient LLM call boundary and provider prerequisite checks."""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableBinding
from pydantic import ConfigDict

from agents.config import Role, Settings
from observability.context import current_agent, record_call
from observability.costs import CallRecord, cost_usd, usage_from_message
from observability.logs import log_event
from resilience import runtime as resilience_runtime
from resilience.breaker import BreakerState
from resilience.cache import ResponseCache
from resilience.errors import classify_llm_error
from resilience.retry import call_with_retry

_EPHEMERAL = {"type": "ephemeral"}
# Models that reject a forced tool_choice ("any"/"tool"/a tool name) with a 400
_NO_FORCED_CHOICE = ("claude-fable-5-1", "claude-opus-5-5", "claude-sonnet-5-5")


def _chat_anthropic() -> type | None:
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:  # pragma: no cover - the provider package is optional
        return None
    return ChatAnthropic


def _supports_forced_tool_choice(model: str) -> bool:
    try:
        from langchain_anthropic.chat_models import _supports_forced_tool_choice as supports
    except ImportError:  # pragma: no cover - private helper moved or renamed
        return not model.startswith(_NO_FORCED_CHOICE)
    return supports(model)


def _is_forced(tool_choice: Any) -> bool:
    if isinstance(tool_choice, dict):
        return tool_choice.get("type") in ("any", "tool")
    return isinstance(tool_choice, str) and tool_choice not in ("auto", "none")


def _with_system_cache_control(messages: list[BaseMessage]) -> list[BaseMessage]:
    """A copy of messages whose first SystemMessage ends in a cache_control breakpoint."""
    out = list(messages)
    for i, m in enumerate(out):
        if not isinstance(m, SystemMessage):
            continue
        if isinstance(m.content, str):
            content: Any = [{"type": "text", "text": m.content, "cache_control": _EPHEMERAL}]
        elif m.content and isinstance(m.content[-1], dict):
            content = [*m.content[:-1], {**m.content[-1], "cache_control": _EPHEMERAL}]
        else:
            return out
        out[i] = m.model_copy(update={"content": content})
        return out
    return out


class ResilientChatModel(BaseChatModel):
    """Delegating wrapper: role breaker outside call-level retry, response cache, cost records and
    Anthropic prompt caching around every call to `inner`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    inner: BaseChatModel
    role: Role
    model_id: str
    cache: ResponseCache | None = None  # shadows BaseChatModel.cache; see _generate_with_cache
    bound: dict = {}
    cacheable: bool = False

    @property
    def _llm_type(self) -> str:
        return "resilient"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> ResilientChatModel:
        """Let the inner model format tools and tool_choice; keep its bound kwargs so calls still pass here."""
        anthropic = _chat_anthropic()
        if (anthropic is not None and isinstance(self.inner, anthropic)
                and not _supports_forced_tool_choice(self.inner.model) and _is_forced(kwargs.get("tool_choice"))):
            kwargs.pop("tool_choice")  # these models 400 on a forced choice; the prompt still asks for the tool
        out = self.inner.bind_tools(tools, **kwargs)
        bound = {**self.bound, **out.kwargs} if isinstance(out, RunnableBinding) else self.bound
        return self.model_copy(update={"bound": bound})

    def cached(self) -> ResilientChatModel:
        """A copy whose calls use the response cache: only for side-effect-free, run-stable calls."""
        return self.model_copy(update={"cacheable": True})

    # BaseChatModel's own LLM-cache step treats a truthy `cache` that isn't a BaseCache as "use the
    # global langchain cache" and raises when none is set. Our cache is handled in _generate instead.
    def _generate_with_cache(self, messages: list[BaseMessage], stop: list[str] | None = None,
                             run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> ChatResult:
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate_with_cache(self, messages: list[BaseMessage], stop: list[str] | None = None,
                                    run_manager: AsyncCallbackManagerForLLMRun | None = None,
                                    **kwargs: Any) -> ChatResult:
        return await self._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def _record(self, msg: AIMessage, *, cache_hit: bool, duration_ms: float, attempts: int) -> None:
        tokens = (0, 0, 0, 0) if cache_hit else usage_from_message(msg)
        cost, priced = cost_usd(self.model_id, *tokens)
        record_call(CallRecord(
            ts=datetime.now(UTC).isoformat(), role=self.role, agent=current_agent() or self.role,
            model=self.model_id, prompt_tokens=tokens[0], completion_tokens=tokens[1],
            cache_read_tokens=tokens[2], cache_write_tokens=tokens[3], cache_hit=cache_hit,
            cost_usd=cost, priced=priced, duration_ms=duration_ms, attempts=attempts))
        log_event("llm_call", role=self.role, model=self.model_id, prompt_tokens=tokens[0],
                  completion_tokens=tokens[1], cache_read_tokens=tokens[2], cache_write_tokens=tokens[3],
                  cost_usd=cost, cache_hit=cache_hit, duration_ms=duration_ms, attempts=attempts)

    def _cache_key(self, messages: list[BaseMessage], fwd: dict, stop: list[str] | None) -> str | None:
        if not (self.cacheable and self.cache is not None and self.cache.enabled):
            return None
        try:
            return self.cache.key(self.model_id, messages, fwd, stop)
        except (TypeError, ValueError) as e:  # an unserializable bound value: just don't cache this call
            log_event("cache_key_failed", level=logging.WARNING, role=self.role, error=str(e))
            return None

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                  run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> ChatResult:
        fwd = {k: v for k, v in {**self.bound, **kwargs}.items() if not k.startswith("ls_")}
        key = self._cache_key(messages, fwd, stop)  # before cache_control, which must not change the key
        rt = resilience_runtime.runtime()
        breaker = rt.breakers.get(self.role)

        if key is not None:
            assert self.cache is not None
            status, cached = self.cache.get(key)
            if status == "hit" and cached is not None:
                degraded = breaker.state is BreakerState.OPEN
                log_event("degraded_fallback_cached" if degraded else "cache_hit",
                          level=logging.WARNING if degraded else logging.INFO, role=self.role, model=self.model_id)
                self._record(cached, cache_hit=True, duration_ms=0.0, attempts=0)
                return ChatResult(generations=[ChatGeneration(message=cached)])
            log_event(f"cache_{status}", role=self.role, model=self.model_id)

        anthropic = _chat_anthropic()
        if self.model_id.startswith("anthropic:") and anthropic is not None and isinstance(self.inner, anthropic):
            messages = _with_system_cache_control(messages)
            tools = fwd.get("tools")
            if tools and isinstance(tools[-1], dict):
                fwd["tools"] = [*tools[:-1], {**tools[-1], "cache_control": _EPHEMERAL}]

        start = time.monotonic()
        result, attempts = breaker.call(lambda: call_with_retry(
            lambda: self.inner._generate(messages, stop=stop, run_manager=None, **fwd),
            policy=rt.policy, classify=classify_llm_error, name=f"llm.{self.role}", sleep=rt.sleep))
        duration_ms = (time.monotonic() - start) * 1000
        msg = result.generations[0].message
        self._record(msg, cache_hit=False, duration_ms=duration_ms, attempts=attempts)

        if key is not None and isinstance(msg, AIMessage):
            assert self.cache is not None
            try:
                self.cache.put(key, msg)
            except Exception as e:  # a cache write never fails the call
                log_event("cache_write_failed", level=logging.WARNING, role=self.role,
                          error=f"{type(e).__name__}: {e}")
        return result


def resilient(model: BaseChatModel, settings: Settings, role: Role, model_id: str | None = None) -> ResilientChatModel:
    """Wrap model (once) for role; also applies settings to the process-wide retry/breaker runtime."""
    if isinstance(model, ResilientChatModel):
        return model
    resilience_runtime.configure(settings)
    return ResilientChatModel(inner=model, role=role, model_id=model_id or settings.model_for(role),
                              cache=ResponseCache(settings.cache_dir, settings.cache_ttl_s))


def make_model(settings: Settings, role: Role = "coder") -> ResilientChatModel:
    model = settings.model_for(role)
    provider = model.partition(":")[0]
    # Our wrapper is the only retry layer, so its attempt counts are honest
    sdk = {"max_retries": 0} if provider in {"anthropic", "openai"} else {}
    return resilient(init_chat_model(model, max_tokens=settings.max_tokens, **sdk), settings, role)


def provider_prerequisite(model: str) -> str | None:
    """None when `model` can be called here, else a reason naming what's missing."""
    provider, _, name = model.partition(":")
    if provider == "anthropic":
        return None if os.environ.get("ANTHROPIC_API_KEY") else f"ANTHROPIC_API_KEY is not set (needed for {model})"
    if provider == "openai":
        return None if os.environ.get("OPENAI_API_KEY") else f"OPENAI_API_KEY is not set (needed for {model})"
    if provider == "ollama":
        base = os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434"
        try:
            with urllib.request.urlopen(f"{base}/api/tags", timeout=2) as resp:
                tags = json.load(resp)
        except OSError as e:
            return f"Ollama server not reachable at {base} ({e}) (needed for {model})"
        names = {m.get("name", "") for m in tags.get("models", [])}
        if name not in names and f"{name}:latest" not in names:
            return f"Ollama model {name!r} is not pulled (run: ollama pull {name})"
        return None
    return f"unsupported provider prefix {provider!r} in {model!r} (use anthropic:, openai: or ollama:)"

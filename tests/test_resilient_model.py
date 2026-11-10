"""ResilientChatModel: retry, role breaker, response cache, cost records and prompt caching at the LLM boundary."""

from __future__ import annotations

import logging

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool

from agents.coder_agent import CoderAgent
from agents.config import Settings
from agents.model import ResilientChatModel, make_model, resilient
from observability.context import RunContext, agent_scope, run_scope
from orchestration.project_state import Spec
from resilience.breaker import BreakerState
from resilience.cache import ResponseCache
from resilience.errors import CircuitOpenError, TransientError
from resilience.runtime import runtime
from tests.fakes import FakeEmbedding, ScriptedChatModel, ai, happy, make_spec, spec_response, tool_call
from tools.specs import TOOL_SPECS


class Flaky(Exception):
    status_code = 429


@pytest.fixture
def settings_with_cache(tmp_path) -> Settings:
    return Settings.from_env(project_dir=tmp_path, cache_ttl_s=60)


def wrap(settings, *responses, role="pm"):
    return resilient(ScriptedChatModel(responses=list(responses)), settings, role)


def structured_call(m, text):
    return m.cached().with_structured_output(Spec, include_raw=True).invoke([HumanMessage(text)])


def events(caplog, name):
    return [r for r in caplog.records if r.getMessage() == name]


def test_transient_llm_error_retried_once_then_succeeds(settings):
    m = wrap(settings, Flaky("429"), ai("hello"))
    assert m.invoke([HumanMessage("x")]).content == "hello"


def test_retry_does_not_replay_agent_loop(settings):
    inner = ScriptedChatModel(responses=[Flaky("1"), *happy()])
    a = CoderAgent(settings, model=inner, embedding_function=FakeEmbedding())
    assert a.run("t").status == "success"
    assert len(inner.calls) == 3  # 1 failed attempt + 2 turns: the retry repeated one call, not the loop


def test_persistent_failure_raises_after_3_retries_and_counts_toward_breaker(settings):
    inner = ScriptedChatModel(responses=[Flaky("x")] * 4)
    with pytest.raises(TransientError):
        resilient(inner, settings, "pm").invoke([HumanMessage("x")])
    assert len(inner.calls) == 4 and runtime().breakers.get("pm").failures == 1


def test_permanent_error_not_retried_and_keeps_its_text(settings):
    inner = ScriptedChatModel(responses=[RuntimeError("boom"), ai("unused")])
    with pytest.raises(Exception, match="boom"):
        resilient(inner, settings, "pm").invoke([HumanMessage("x")])
    assert len(inner.calls) == 1 and runtime().breakers.get("pm").failures == 0


def test_structured_output_with_include_raw_through_wrapper(settings):
    m = wrap(settings, spec_response(make_spec()))
    out = structured_call(m, "req")
    assert out["parsing_error"] is None and isinstance(out["parsed"], Spec)


def test_bind_tools_delegates_formatting_to_inner(settings, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    from langchain_anthropic import ChatAnthropic
    m = resilient(ChatAnthropic(model="claude-haiku-5-5", max_retries=0), settings, "coder")
    b = m.bind_tools(TOOL_SPECS)
    assert isinstance(b, ResilientChatModel) and b.bound["tools"][0]["name"] == TOOL_SPECS[0]["name"]
    assert "input_schema" in b.bound["tools"][0]  # Anthropic's own tool format, not ours
    assert m.bound == {}  # bind_tools returns a copy


def _structured_binding(structured) -> ResilientChatModel:
    """The ResilientChatModel inside with_structured_output's RunnableMap(raw=llm) | parser."""
    return structured.first.steps__["raw"]


def test_no_forced_tool_choice_for_models_that_reject_it(settings, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    from langchain_anthropic import ChatAnthropic
    sonnet = resilient(ChatAnthropic(model="claude-sonnet-5-5", max_retries=0), settings, "pm")
    bound = _structured_binding(sonnet.with_structured_output(Spec, include_raw=True)).bound
    assert bound["tools"][0]["name"] == "Spec"
    assert bound.get("tool_choice", {}).get("type") not in ("any", "tool")
    haiku = resilient(ChatAnthropic(model="claude-haiku-5-5", max_retries=0), settings, "qa")
    assert _structured_binding(haiku.with_structured_output(Spec, include_raw=True)).bound["tool_choice"] == {
        "type": "any"}


def test_cache_hit_within_ttl_records_zero_cost(settings_with_cache, caplog):
    caplog.set_level(logging.INFO, logger="mad")
    ctx = RunContext(run_id="r")
    with run_scope(ctx):
        m = wrap(settings_with_cache, spec_response(make_spec()))
        for _ in range(2):
            out = structured_call(m, "same")
    assert isinstance(out["parsed"], Spec)
    assert [c.cache_hit for c in ctx.calls] == [False, True] and ctx.calls[1].cost_usd == 0
    assert ctx.calls[1].prompt_tokens == 0 and ctx.calls[1].completion_tokens == 0
    assert len(events(caplog, "cache_miss")) == 1 and len(events(caplog, "cache_hit")) == 1


def test_expired_entry_is_refetched(settings_with_cache, caplog):
    caplog.set_level(logging.INFO, logger="mad")
    m = wrap(settings_with_cache, spec_response(make_spec()), spec_response(make_spec()))
    structured_call(m, "same")
    m.response_cache.clock = lambda: 10**12  # far past the TTL
    structured_call(m, "same")
    assert len(m.inner.calls) == 2 and len(events(caplog, "cache_expired")) == 1


class ToolBindingScripted(ScriptedChatModel):
    """Binds formatted tools like a real provider, so the wrapper sees `tools` in its bound kwargs."""

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=[convert_to_openai_tool(t) for t in tools], **kwargs)


def test_structured_reply_without_tool_call_is_not_cached(settings_with_cache, caplog):
    caplog.set_level(logging.INFO, logger="mad")
    m = resilient(ToolBindingScripted(responses=[ai("not a spec"), ai("not a spec")]), settings_with_cache, "pm")
    for _ in range(2):
        out = structured_call(m, "same")
        assert out["parsed"] is None
    assert len(m.inner.calls) == 2 and len(events(caplog, "cache_miss")) == 2
    assert not events(caplog, "cache_hit")
    cache_dir = settings_with_cache.cache_dir
    assert not cache_dir.exists() or not [p for p in cache_dir.rglob("*") if p.is_file()]


def test_reply_cut_off_at_max_tokens_is_not_cached(settings_with_cache, caplog):
    caplog.set_level(logging.INFO, logger="mad")
    cut = spec_response(make_spec())
    cut.response_metadata = {"stop_reason": "max_tokens"}
    m = wrap(settings_with_cache, cut, spec_response(make_spec()))
    for _ in range(2):
        structured_call(m, "same")
    assert len(m.inner.calls) == 2 and not events(caplog, "cache_hit")


def test_uncached_calls_never_hit_cache(settings_with_cache):
    ctx = RunContext(run_id="r")
    inner = ScriptedChatModel(responses=[spec_response(make_spec()), spec_response(make_spec())])
    with run_scope(ctx):
        m = resilient(inner, settings_with_cache, "pm")
        for _ in range(2):
            m.with_structured_output(Spec, include_raw=True).invoke([HumanMessage("same")])
    assert len(inner.calls) == 2 and not inner.responses
    assert [c.cache_hit for c in ctx.calls] == [False, False]


def test_open_circuit_returns_cached_or_structured_error(settings_with_cache, caplog):
    caplog.set_level(logging.INFO, logger="mad")
    inner = ScriptedChatModel(responses=[spec_response(make_spec())])
    m = resilient(inner, settings_with_cache, "pm")
    structured_call(m, "warm")  # 1. warm the cache
    breaker = runtime().breakers.get("pm")

    def fail():
        raise TransientError("down")

    for _ in range(3):  # 2. force the pm breaker OPEN
        with pytest.raises(TransientError):
            breaker.call(fail)
    assert breaker.state is BreakerState.OPEN

    out = structured_call(m, "warm")  # 3. served from the cache, flagged as degraded
    assert isinstance(out["parsed"], Spec) and len(events(caplog, "degraded_fallback_cached")) == 1

    with pytest.raises(CircuitOpenError):  # 4. no entry: a structured error, inner never called
        structured_call(m, "a different prompt")
    assert len(inner.calls) == 1


def test_usage_recorded_after_every_call(settings):
    msg = AIMessage("ok", usage_metadata={"input_tokens": 120, "output_tokens": 30, "total_tokens": 150,
                                          "input_token_details": {"cache_read": 100, "cache_creation": 10}})
    ctx = RunContext(run_id="r")
    with run_scope(ctx), agent_scope("pm-consolidate"):
        resilient(ScriptedChatModel(responses=[Flaky("1"), msg]), settings, "pm").invoke([HumanMessage("x")])
    (c,) = ctx.calls
    assert (c.prompt_tokens, c.completion_tokens, c.cache_read_tokens, c.cache_write_tokens) == (120, 30, 100, 10)
    assert c.role == "pm" and c.agent == "pm-consolidate" and c.model == settings.pm_model
    assert c.attempts == 2 and c.cost_usd > 0 and c.priced and not c.cache_hit


def test_agent_defaults_to_role(settings):
    ctx = RunContext(run_id="r")
    with run_scope(ctx):
        wrap(settings, ai("ok"), role="qa").invoke([HumanMessage("x")])
    assert ctx.calls[0].agent == "qa" and ctx.calls[0].model == settings.qa_model


def test_cache_write_failure_never_fails_the_call(settings_with_cache, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="mad")
    m = wrap(settings_with_cache, spec_response(make_spec()))

    def broken_put(key, message):
        raise OSError("disk full")

    monkeypatch.setattr(m.response_cache, "put", broken_put)
    assert isinstance(structured_call(m, "x")["parsed"], Spec)
    assert len(events(caplog, "cache_write_failed")) == 1


def test_unkeyable_bound_is_treated_as_not_cacheable(settings_with_cache):
    inner = ScriptedChatModel(responses=[ai("a"), ai("b")])
    m = resilient(inner, settings_with_cache, "pm").model_copy(update={"bound": {"weird": object()}}).cached()
    assert [m.invoke([HumanMessage("x")]).content for _ in range(2)] == ["a", "b"]


def test_ls_kwargs_are_not_forwarded(settings):
    seen = []

    class Recording(ScriptedChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            seen.append(kwargs)
            return super()._generate(messages, stop, run_manager, **kwargs)

    m = resilient(Recording(responses=[ai("ok")]), settings, "pm")
    m.model_copy(update={"bound": {"ls_structured_output_format": {}, "temperature": 0}}).invoke([HumanMessage("x")])
    assert seen == [{"temperature": 0}]


def test_anthropic_prompt_caching_marks_system_prompt_and_last_tool(settings_with_cache, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    from langchain_anthropic import ChatAnthropic

    payloads = []

    class Offline(ChatAnthropic):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            payloads.append(self._get_request_payload(messages, stop=stop, **kwargs))
            # A tools-bound reply must carry a tool call to be cached
            return ChatResult(generations=[ChatGeneration(message=ai("ok", tool_call("finish", explanation="x")))])

    # model_id explicit: a leaked CODER_MODEL from another test must not turn prompt caching off
    m = resilient(Offline(model="claude-haiku-5-5", max_retries=0), settings_with_cache, "coder",
                  model_id="anthropic:claude-haiku-5-5")
    b = m.bind_tools(TOOL_SPECS).cached()
    system = SystemMessage("You are the coder.")
    msgs = [system, HumanMessage("task")]
    b.invoke(msgs)
    (p,) = payloads
    assert p["system"] == [{"type": "text", "text": "You are the coder.", "cache_control": {"type": "ephemeral"}}]
    assert p["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert all("cache_control" not in t for t in p["tools"][:-1])
    assert system.content == "You are the coder." and "cache_control" not in b.bound["tools"][-1]  # caller's copies
    # the response-cache key is computed before cache_control is added
    key = b.response_cache.key(b.model_id, msgs, b.bound, None)
    assert b.response_cache.get(key)[0] == "hit"


def test_no_prompt_caching_for_non_anthropic_models(settings):
    seen = []

    class Recording(ScriptedChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            seen.append(messages)
            return super()._generate(messages, stop, run_manager, **kwargs)

    resilient(Recording(responses=[ai("ok")]), settings, "pm").invoke([SystemMessage("sys"), HumanMessage("x")])
    assert seen[0][0].content == "sys"


def test_langchain_cache_stays_off(settings):
    from langchain_core.globals import get_llm_cache

    m = wrap(settings, ai("x"))
    assert m.cache is None and get_llm_cache() is None  # base _generate_with_cache goes straight to _generate
    assert isinstance(m.response_cache, ResponseCache)


def test_resilient_is_idempotent(settings):
    m = wrap(settings, ai("x"))
    assert resilient(m, settings, "pm") is m


def test_make_model_uses_role_model_and_disables_sdk_retries(settings, monkeypatch):
    captured = {}

    def fake_init(model, **kwargs):
        captured.clear()
        captured.update(model=model, **kwargs)
        return ScriptedChatModel(responses=[])

    monkeypatch.setattr("agents.model.init_chat_model", fake_init)
    m = make_model(settings, "pm")
    assert isinstance(m, ResilientChatModel) and m.role == "pm" and m.model_id == settings.model_for("pm")
    assert captured == {"model": settings.model_for("pm"), "max_tokens": settings.max_tokens, "max_retries": 0}
    make_model(settings.model_copy(update={"qa_max_tokens": 16000}), "qa")
    assert captured["max_tokens"] == 16000  # QA writes whole test modules: its own output budget
    make_model(settings.model_copy(update={"model": "ollama:llama3"}), "coder")
    assert captured["model"] == "ollama:llama3" and "max_retries" not in captured


def test_agents_wrap_injected_models_with_their_role(settings):
    from agents.pm_agent import PMAgent
    from agents.qa_agent import QAAgent

    inner = ScriptedChatModel(responses=[])
    for agent_cls, role in ((PMAgent, "pm"), (QAAgent, "qa")):
        a = agent_cls(settings, model=inner)
        assert isinstance(a.model, ResilientChatModel) and a.model.role == role and a.model.inner is inner
    c = CoderAgent(settings, model=inner, embedding_function=FakeEmbedding())
    assert c.model.role == "coder" and c.model.inner is inner


def test_trace_params_report_the_inner_model(settings, monkeypatch):
    from langchain_anthropic import ChatAnthropic

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")
    inner = ChatAnthropic(model="claude-haiku-5-5", max_retries=0, api_key="sk-ant-dummy")
    params = resilient(inner, settings, "coder", "anthropic:claude-haiku-5-5")._get_ls_params()
    assert params["ls_model_name"] == "claude-haiku-5-5" and params["ls_provider"] == "anthropic"


def test_trace_params_fall_back_when_inner_raises(settings, monkeypatch):
    inner = ScriptedChatModel(responses=[])

    def boom(*a, **k):
        raise RuntimeError("no params")

    object.__setattr__(inner, "_get_ls_params", boom)
    params = resilient(inner, settings, "pm")._get_ls_params()
    assert params["ls_model_type"] == "chat"

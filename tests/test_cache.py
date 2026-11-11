"""On-disk response cache: TTL, key sensitivity, round-trip fidelity, corrupt files."""

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

import resilience.cache as cache_mod
from resilience.cache import ResponseCache


def test_hit_within_ttl(tmp_path):
    t = [1000.0]
    c = ResponseCache(tmp_path, ttl_s=60, clock=lambda: t[0])
    k = c.key("m", [HumanMessage("hi")], {}, None)
    assert c.get(k) == ("miss", None)
    c.put(k, AIMessage("cached"))
    t[0] += 59
    status, msg = c.get(k)
    assert status == "hit" and msg.content == "cached"


def test_expired_after_ttl(tmp_path):
    t = [1000.0]
    c = ResponseCache(tmp_path, ttl_s=60, clock=lambda: t[0])
    k = c.key("m", [HumanMessage("hi")], {}, None)
    c.put(k, AIMessage("cached"))
    t[0] += 60.01
    assert c.get(k) == ("expired", None)


def test_key_changes_with_messages_tools_model(tmp_path):
    c = ResponseCache(tmp_path, ttl_s=60)
    msgs = [HumanMessage("hi")]
    base = c.key("m", msgs, {"tools": ["a"]}, None)
    assert base == c.key("m", [HumanMessage("hi")], {"tools": ["a"]}, None)
    assert len({base,
                c.key("m", [HumanMessage("other")], {"tools": ["a"]}, None),
                c.key("m", msgs, {"tools": ["b"]}, None),
                c.key("m2", msgs, {"tools": ["a"]}, None)}) == 4


def test_disabled_when_ttl_zero(tmp_path):
    c = ResponseCache(tmp_path, 0)
    assert c.enabled is False
    k = c.key("m", [HumanMessage("hi")], {}, None)
    c.put(k, AIMessage("x"))
    assert c.get(k) == ("miss", None)
    assert ResponseCache(tmp_path, 1).enabled is True


def test_round_trip_preserves_tool_calls_and_usage(tmp_path):
    c = ResponseCache(tmp_path, ttl_s=60)
    k = c.key("m", [HumanMessage("hi")], {"tool_choice": "any"}, ["x"])
    msg = AIMessage("", tool_calls=[{"name": "Spec", "args": {"a": 1}, "id": "c1", "type": "tool_call"}],
                    usage_metadata={"input_tokens": 3, "output_tokens": 4, "total_tokens": 7})
    c.put(k, msg)
    status, got = c.get(k)
    assert status == "hit"
    assert got.tool_calls == msg.tool_calls
    assert got.usage_metadata["total_tokens"] == 7


def test_corrupt_or_partial_file_is_a_miss(tmp_path):
    c = ResponseCache(tmp_path, ttl_s=60)
    k = c.key("m", [HumanMessage("hi")], {}, None)
    c.put(k, AIMessage("ok"))
    path = next(tmp_path.rglob(f"{k}.json"))
    path.write_text('{"created_at": 1, "mess')
    assert c.get(k) == ("miss", None)
    c.put(k, AIMessage("again"))
    assert c.get(k)[0] == "hit"


def test_wrong_shape_file_is_a_miss(tmp_path):
    c = ResponseCache(tmp_path, ttl_s=60)
    k = c.key("m", [HumanMessage("hi")], {}, None)
    c.put(k, AIMessage("ok"))
    path = next(tmp_path.rglob(f"{k}.json"))
    path.write_text('{"created_at": "x", "message": 5}')
    assert c.get(k) == ("miss", None)
    path.write_text('[1]')
    assert c.get(k) == ("miss", None)


def test_put_leaves_no_temp_files(tmp_path):
    c = ResponseCache(tmp_path, ttl_s=60)
    k = c.key("m", [HumanMessage("hi")], {}, None)
    c.put(k, AIMessage("ok"))
    assert [p.name for p in tmp_path.rglob("*") if p.is_file()] == [f"{k}.json"]


def _hist(ai_id, call_id, usage):
    return [HumanMessage("hi", id=ai_id + "h"),
            AIMessage("", id=ai_id,
                      tool_calls=[{"name": "Spec", "args": {"a": 1}, "id": call_id, "type": "tool_call"}],
                      response_metadata={"x": call_id}, usage_metadata=usage),
            ToolMessage("done", tool_call_id=call_id, id=ai_id + "t")]


def test_key_ignores_random_ids_and_metadata(tmp_path):
    c = ResponseCache(tmp_path, ttl_s=60)
    u1 = {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}
    u2 = {"input_tokens": 9, "output_tokens": 9, "total_tokens": 18}
    assert c.key("m", _hist("lc_run--aaa-0", "toolu_1", u1), {}, None) == \
        c.key("m", _hist("lc_run--bbb-0", "toolu_2", u2), {}, None)
    plain = [HumanMessage("hi"), AIMessage("x"), ToolMessage("done", tool_call_id="q")]
    assert c.key("m", plain, {}, None) == c.key(
        "m",
        [HumanMessage("hi", id="r"), AIMessage("x", id="s"), ToolMessage("done", tool_call_id="z", id="y")],
        {},
        None,
    )


def test_key_still_depends_on_tool_call_name_and_args(tmp_path):
    c = ResponseCache(tmp_path, ttl_s=60)
    a = _hist("i", "c", None)
    b = _hist("i", "c", None)
    b[1].tool_calls[0]["args"] = {"a": 2}
    assert c.key("m", a, {}, None) != c.key("m", b, {}, None)


class _Schema(BaseModel):
    x: int


def _tool_fn(x):
    return x


def test_bound_class_and_function_keys_are_stable(tmp_path, monkeypatch):
    c = ResponseCache(tmp_path, ttl_s=60)
    seen = []
    real = json.dumps
    monkeypatch.setattr(cache_mod.json, "dumps", lambda *a, **k: seen.append(real(*a, **k)) or seen[-1])
    bound = {"tools": [_Schema, _tool_fn], "tool_choice": "any"}
    k1 = c.key("m", [HumanMessage("hi")], bound, None)
    k2 = c.key("m", [HumanMessage("hi")], {"tools": [_Schema, _tool_fn], "tool_choice": "any"}, None)
    assert k1 == k2
    assert "0x" not in seen[0]
    assert f"{__name__}._Schema" in seen[0] and f"{__name__}._tool_fn" in seen[0]


def test_unserializable_bound_object_raises(tmp_path):
    import pytest
    with pytest.raises(TypeError):
        ResponseCache(tmp_path, ttl_s=60).key("m", [], {"x": object()}, None)


def test_key_ignores_tool_use_ids_in_list_content(tmp_path):
    c = ResponseCache(tmp_path, ttl_s=60)

    def msgs(tid, inp):
        return [AIMessage([{"type": "text", "text": "go"},
                           {"type": "tool_use", "id": tid, "name": "Spec", "input": inp}]),
                ToolMessage([{"type": "tool_result", "tool_use_id": tid, "content": "ok"}],
                            tool_call_id=tid)]

    a = msgs("toolu_1", {"a": 1})
    assert c.key("m", a, {}, None) == c.key("m", msgs("toolu_2", {"a": 1}), {}, None)
    assert c.key("m", a, {}, None) != c.key("m", msgs("toolu_1", {"a": 2}), {}, None)
    assert a[0].content[1]["id"] == "toolu_1"  # inputs not mutated

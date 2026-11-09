"""On-disk response cache: TTL, key sensitivity, round-trip fidelity, corrupt files."""

from langchain_core.messages import AIMessage, HumanMessage

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
    k = c.key("m", [HumanMessage("hi")], {"tool_choice": object()}, ["x"])
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

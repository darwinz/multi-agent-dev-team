from orchestration.routing import classify, extract_plan
from tests.fakes import ai, tool_call

W = tool_call("write_file", path="a.py", content="x")
F = tool_call("finish", explanation="e", plan=["p"])


def test_finish_alone():
    d = classify(ai("", F), 3, 10)
    assert d.kind == "finish" and d.finish_args["plan"] == ["p"]


def test_finish_alone_at_limit_is_honoured():
    assert classify(ai("", F), 10, 10).kind == "finish"


def test_finish_mixed_is_rejected_others_run():
    d = classify(ai("", W, F), 3, 10)
    assert d.kind == "execute" and [c["name"] for c in d.run_calls] == ["write_file"]
    assert d.rejections == [(F["id"], "finish must be called alone, after your last action")]


def test_tools_at_limit_not_executed():
    d = classify(ai("", W), 10, 10)
    assert d.kind == "limit" and d.run_calls == []


def test_text_only_nudges_then_limits():
    assert classify(ai("thinking"), 4, 10).kind == "nudge"
    assert classify(ai("thinking"), 10, 10).kind == "limit"


def test_invalid_finish_args_is_case_4():
    bad = tool_call("finish", explanation="e")  # missing plan
    d = classify(ai("", bad), 2, 10)
    assert d.kind == "nudge" and d.rejections[0][1].startswith("finish rejected:")


def test_extract_plan():
    assert extract_plan("Plan:\n1. write\n2) test\nok") == ["write", "test"]

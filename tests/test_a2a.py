import pytest

from orchestration.a2a import (
    AGENTS,
    CODER_ACCEPTS,
    INTENTS,
    QA_ACCEPTS,
    A2AError,
    Approved,
    Broker,
    Endpoint,
    FinalReport,
    FixInstruction,
    Message,
    ReviewRequest,
)
from orchestration.review import Issue

ISSUE = Issue(severity="High", test="t", criterion="c", location="l", assertion="a", fix="f")


def review(task_id="T1"):
    return ReviewRequest(task_id=task_id, round=1, files={"a.py": "h"})


def raw(sender, receiver, intent, payload, **kw):
    return Message(sender=sender, receiver=receiver, intent=intent, payload=payload, **kw)


def test_message_has_five_fields():
    assert set(Message.model_fields) == {"sender", "receiver", "intent", "payload", "correlation_id"}


def test_constants():
    assert AGENTS == {"coder", "qa"}
    assert set(INTENTS) == {"review_request", "fix_instruction", "approved", "final_report"}
    assert QA_ACCEPTS == {("coder", "review_request")}
    assert CODER_ACCEPTS == {("qa", "fix_instruction"), ("qa", "approved"), ("qa", "final_report")}


def test_message_rejects_unknown_agent_intent_and_self_send():
    p = review().model_dump()
    with pytest.raises(ValueError):
        raw("coder", "pm", "review_request", p)
    with pytest.raises(ValueError):
        raw("coder", "qa", "gossip", p)
    with pytest.raises(ValueError):
        raw("coder", "coder", "review_request", p)


def test_message_is_frozen():
    m = raw("coder", "qa", "review_request", review().model_dump())
    with pytest.raises(ValueError):
        m.intent = "approved"


def test_payload_validated_per_intent():
    with pytest.raises(ValueError):
        raw("coder", "qa", "review_request", {"task_id": "T1", "round": 1})
    with pytest.raises(ValueError):
        raw("qa", "coder", "fix_instruction", {"task_id": "T1", "round": 1, "issues": [], "summary": "s"})
    with pytest.raises(ValueError):
        raw("coder", "qa", "review_request", {**review().model_dump(), "round": 0})


def test_correlation_ids_unique():
    p = review().model_dump()
    assert raw("coder", "qa", "review_request", p).correlation_id != raw("coder", "qa", "review_request", p).correlation_id


def test_broker_fifo_per_agent():
    b = Broker()
    m1 = raw("coder", "qa", "review_request", review("T1").model_dump())
    m2 = raw("coder", "qa", "review_request", review("T2").model_dump())
    b.send(m1)
    b.send(m2)
    assert b.pending("qa") == 2 and b.pending("coder") == 0
    assert b.receive("qa") is m1
    assert b.receive("qa") is m2
    assert b.pending("qa") == 0


def test_receive_empty_raises():
    with pytest.raises(A2AError):
        Broker().receive("qa")


def test_on_event_records_sent_and_received():
    events = []
    b = Broker(on_event=events.append)
    m = raw("coder", "qa", "review_request", review().model_dump())
    b.send(m)
    b.receive("qa")
    assert [e["event"] for e in events] == ["sent", "received"]
    for e in events:
        assert set(e) == {"event", "message", "reason", "at"}
        assert e["message"] == m.model_dump(mode="json")
        assert e["reason"] is None


def endpoints(events=None):
    b = Broker(on_event=events.append if events is not None else None)
    return b, Endpoint("coder", CODER_ACCEPTS, b), Endpoint("qa", QA_ACCEPTS, b)


def test_endpoint_send_builds_message():
    b, coder, _ = endpoints()
    m = coder.send("qa", "review_request", review())
    assert m.sender == "coder" and m.payload["task_id"] == "T1"
    assert b.pending("qa") == 1
    assert coder.send("qa", "review_request", review(), correlation_id="abc").correlation_id == "abc"


def test_endpoint_discards_unaccepted_intent():
    events = []
    b, coder, qa = endpoints(events)
    b.send(raw("coder", "qa", "fix_instruction", FixInstruction(task_id="T1", round=1, issues=[ISSUE], summary="s").model_dump()))
    real = coder.send("qa", "review_request", review())
    assert qa.receive(task_id="T1") == real
    rejected = [e for e in events if e["event"] == "rejected"]
    assert len(rejected) == 1
    assert (rejected[0]["message"]["sender"], rejected[0]["message"]["intent"]) == ("coder", "fix_instruction")
    assert rejected[0]["reason"]


def test_endpoint_discards_wrong_task():
    events = []
    _, coder, qa = endpoints(events)
    coder.send("qa", "review_request", review("T9"))
    good = coder.send("qa", "review_request", review("T1"))
    assert qa.receive(task_id="T1") == good
    assert [e["event"] for e in events].count("rejected") == 1


def test_coder_discards_mismatched_correlation():
    events = []
    _, coder, qa = endpoints(events)
    qa.send("coder", "approved", Approved(task_id="T1", round=1, tests_run=2, test_file="test_qa_t1.py"), correlation_id="stale")
    good = qa.send("coder", "approved", Approved(task_id="T1", round=1, tests_run=2, test_file="test_qa_t1.py"), correlation_id="live")
    assert coder.receive(task_id="T1", correlation_id="live") == good
    assert [e["event"] for e in events].count("rejected") == 1


def test_qa_does_not_check_correlation():
    _, coder, qa = endpoints()
    m = coder.send("qa", "review_request", review())
    assert qa.receive(task_id="T1", correlation_id="whatever") == m


def test_endpoint_exhausted_raises():
    _, coder, qa = endpoints()
    coder.send("qa", "review_request", review("T9"))
    with pytest.raises(A2AError):
        qa.receive(task_id="T1")


def test_final_report_payload_roundtrip():
    _, coder, qa = endpoints()
    qa.send("coder", "final_report", FinalReport(task_id="T1", round=3, reason="max_iterations", unresolved=[ISSUE], summary="s"))
    assert coder.receive(task_id="T1").payload["unresolved"][0]["severity"] == "High"

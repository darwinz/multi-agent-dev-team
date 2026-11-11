"""Agent-to-agent protocol: a validated message, an in-process broker and validating endpoints."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from orchestration.review import Issue

AGENTS = frozenset({"coder", "qa"})


class A2AError(Exception):
    """A protocol bug: an empty queue, or no acceptable message. Both ends are ours, so it propagates."""


class ReviewRequest(BaseModel):
    """Coder to QA: please review these files (path to sha256) for the task, round n."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    round: int = Field(ge=1)
    files: dict[str, str] = Field(description="path -> sha256")


class FixInstruction(BaseModel):
    """QA to Coder: the issues to fix for the task this round, with a summary."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    round: int = Field(ge=1)
    issues: list[Issue] = Field(min_length=1)
    summary: str


class Approved(BaseModel):
    """QA to Coder: the task's tests all passed; carries the test count and test file."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    round: int = Field(ge=1)
    tests_run: int = Field(ge=1)
    test_file: str


class FinalReport(BaseModel):
    """QA to Coder: the review loop ended unresolved (reason), listing the open issues."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    round: int = Field(ge=1)
    reason: Literal["max_iterations", "qa_error", "no_revision"]
    unresolved: list[Issue]
    summary: str


INTENTS: dict[str, type[BaseModel]] = {
    "review_request": ReviewRequest,
    "fix_instruction": FixInstruction,
    "approved": Approved,
    "final_report": FinalReport,
}

QA_ACCEPTS: frozenset[tuple[str, str]] = frozenset({("coder", "review_request")})
CODER_ACCEPTS: frozenset[tuple[str, str]] = frozenset(
    {("qa", "fix_instruction"), ("qa", "approved"), ("qa", "final_report")}
)


class Message(BaseModel):
    """A validated, immutable A2A message: known sender, receiver and intent, with a payload matching the intent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sender: str
    receiver: str
    intent: str
    payload: dict
    correlation_id: str = Field(default_factory=lambda: uuid4().hex)

    @model_validator(mode="after")
    def _well_formed(self) -> Message:
        for role, agent in (("sender", self.sender), ("receiver", self.receiver)):
            if agent not in AGENTS:
                raise ValueError(f"unknown {role} {agent!r}; agents are {sorted(AGENTS)}")
        if self.sender == self.receiver:
            raise ValueError(f"{self.sender!r} cannot message itself")
        model = INTENTS.get(self.intent)
        if model is None:
            raise ValueError(f"unknown intent {self.intent!r}; intents are {sorted(INTENTS)}")
        model.model_validate(self.payload)
        return self


class Broker:
    """One FIFO queue per agent. Agents hold only the broker, never each other."""

    def __init__(self, on_event: Callable[[dict], None] | None = None) -> None:
        """Create one empty queue per agent; on_event (optional) gets every sent, received and rejected record."""
        self._queues: dict[str, deque[Message]] = {a: deque() for a in AGENTS}
        self.on_event = on_event

    def _emit(self, event: str, msg: Message, reason: str | None = None) -> None:
        if self.on_event:
            self.on_event({
                "event": event,
                "message": msg.model_dump(mode="json"),
                "reason": reason,
                "at": datetime.now(UTC).isoformat(),
            })

    def send(self, msg: Message) -> None:
        """Queue the message for its receiver and emit a 'sent' event."""
        self._queues[msg.receiver].append(msg)
        self._emit("sent", msg)

    def receive(self, agent: str) -> Message:
        """Pop the agent's oldest message and emit 'received'; raises A2AError when its queue is empty."""
        queue = self._queues[agent]
        if not queue:
            raise A2AError(f"{agent} has no pending messages")
        msg = queue.popleft()
        self._emit("received", msg)
        return msg

    def reject(self, msg: Message, reason: str) -> None:
        """Record a message an endpoint discarded."""
        self._emit("rejected", msg, reason)

    def pending(self, agent: str) -> int:
        """Number of messages waiting for the agent."""
        return len(self._queues[agent])


class Endpoint:
    """An agent's view of the broker. `receive` is the trust boundary: it discards what the agent
    did not ask for."""

    def __init__(self, name: str, accepts: frozenset[tuple[str, str]], broker: Broker) -> None:
        """Bind an agent name, the (sender, intent) pairs it accepts and the broker it talks through."""
        self.name = name
        self.accepts = accepts
        self.broker = broker

    def send(
        self, receiver: str, intent: str, payload: BaseModel, *, correlation_id: str | None = None
    ) -> Message:
        """Build a validated Message from this agent and queue it on the broker; returns it."""
        fields = {"correlation_id": correlation_id} if correlation_id else {}
        msg = Message(
            sender=self.name, receiver=receiver, intent=intent, payload=payload.model_dump(mode="json"), **fields
        )
        self.broker.send(msg)
        return msg

    def receive(self, *, task_id: str, correlation_id: str | None = None) -> Message:
        """The first acceptable message for `task_id`. The Coder also matches `correlation_id`."""
        while True:
            msg = self.broker.receive(self.name)
            if (msg.sender, msg.intent) not in self.accepts:
                reason = f"{self.name} does not accept {msg.intent!r} from {msg.sender!r}"
            elif msg.payload.get("task_id") != task_id:
                reason = f"task {msg.payload.get('task_id')!r} is not the current task {task_id!r}"
            elif self.name == "coder" and correlation_id is not None and msg.correlation_id != correlation_id:
                reason = f"correlation {msg.correlation_id!r} is not the outstanding {correlation_id!r}"
            else:
                return msg
            self.broker.reject(msg, reason)

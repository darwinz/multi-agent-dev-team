"""Test doubles shared across the unit tests."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

_DIMS = 64


class FakeEmbedding(EmbeddingFunction[Documents]):
    """Deterministic bag-of-words hashing, L2-normalised: texts sharing words are close. No download."""

    def __init__(self) -> None:
        pass

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embed(text) for text in input]

    @staticmethod
    def _embed(text: str) -> list[float]:
        vec = [0.0] * _DIMS
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % _DIMS] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    @staticmethod
    def name() -> str:
        return "fake-bag-of-words"

    def get_config(self) -> dict[str, Any]:
        return {}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> FakeEmbedding:
        return FakeEmbedding()


# --- scripted chat model -----------------------------------------------------------

import copy  # noqa: E402
import uuid  # noqa: E402

from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, BaseMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402


def tool_call(name: str, **args: Any) -> dict:
    return {"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:12]}", "type": "tool_call"}


def ai(text: str = "", *calls: dict) -> AIMessage:
    # A unique id per message: add_messages replaces messages that share an id
    return AIMessage(content=text, tool_calls=list(calls), id=f"ai_{uuid.uuid4().hex[:12]}")


class ScriptedChatModel(BaseChatModel):
    """Returns pre-scripted responses in order and records every message list it was sent."""

    responses: list[Any]
    calls: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:
        return self

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        self.calls.append(list(messages))
        if not self.responses:
            raise AssertionError("model called more times than scripted")
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return ChatResult(generations=[ChatGeneration(message=copy.deepcopy(nxt))])


PAL = "def is_pal(s):\n    return s == s[::-1]\n"
TEST = "from palindrome import is_pal\n\ndef test_it():\n    assert is_pal('aba')\n"


def happy() -> list[AIMessage]:
    """A fresh two-response script: write module + test and run it, then finish."""
    return [
        ai("1. write module\n2. write test\n3. run test",
           tool_call("write_file", path="palindrome.py", content=PAL),
           tool_call("write_file", path="test_palindrome.py", content=TEST),
           tool_call("exec_python", path="test_palindrome.py")),
        ai("done", tool_call("finish", explanation="Checks palindromes", plan=["write", "test"],
                             entry_point="test_palindrome.py")),
    ]


PAL_REVISED = 'def is_pal(s):\n    """True if s reads the same backwards."""\n    return s == s[::-1]\n'


def revise_happy() -> list[AIMessage]:
    """The revision run: rewrite the module (docstring added), rerun the test, finish."""
    return [
        ai("apply the self-review",
           tool_call("write_file", path="palindrome.py", content=PAL_REVISED),
           tool_call("exec_python", path="test_palindrome.py")),
        ai("done", tool_call("finish", explanation="Checks palindromes, documented", plan=["revise"],
                             entry_point="test_palindrome.py")),
    ]


def agent(settings: Any, *responses: Any) -> Any:
    from agents.coder_agent import CoderAgent

    return CoderAgent(settings, model=ScriptedChatModel(responses=list(responses)), embedding_function=FakeEmbedding())


# --- project state fixtures -------------------------------------------------------

def make_spec(n_tasks: int = 2, n_criteria: int = 2):
    """A valid Spec: criteria AC1..ACn, tasks T1..Tn. T1 covers every criterion; Ti (i >= 2) covers one."""
    from orchestration.project_state import AcceptanceCriterion, Spec, Task

    ids = [f"AC{i}" for i in range(1, n_criteria + 1)]
    tasks = [
        Task(
            task_id=f"T{i}",
            title=f"Task {i}",
            description=f"Implement part {i} of the demo API.",
            acceptance_criteria=[f"Behaviour {i} works when the code is run"],
            priority="high",
            covers=list(ids) if i == 1 else [f"AC{(i - 2) % n_criteria + 1}"],
        )
        for i in range(1, n_tasks + 1)
    ]
    return Spec(
        name="Demo API",
        description="A small demo API.",
        acceptance_criteria=[AcceptanceCriterion(id=i, text=f"Criterion {i} holds") for i in ids],
        tasks=tasks,
    )


def structured(schema: str, payload: dict) -> AIMessage:
    """A structured-output reply: a tool call named after the schema class. Built directly because
    tool_call(name, **args) cannot carry a payload field called `name`."""
    call = {"name": schema, "args": payload, "id": f"call_{uuid.uuid4().hex[:12]}", "type": "tool_call"}
    return AIMessage(content="", tool_calls=[call], id=f"ai_{uuid.uuid4().hex[:12]}")


def spec_response(spec: Any) -> AIMessage:
    return structured("Spec", spec.model_dump())


# --- QA agent -----------------------------------------------------------------------

QA_PAL_TESTS = (
    "from palindrome import is_pal\n\n\n"
    "def test_pal_qa():\n"
    "    assert is_pal('abba')\n"
    "    assert not is_pal('ab')\n"
)


def qa_suite_response(content: str = QA_PAL_TESTS, tests: tuple[tuple[str, int], ...] = (("test_pal_qa", 1),)) -> AIMessage:
    return structured("QATestSuite", {"content": content, "tests": [{"name": n, "criterion": c} for n, c in tests]})


def fix_plan_response(*fixes: tuple[str, str, str]) -> AIMessage:
    return structured("FixPlan", {"fixes": [{"test": t, "root_cause": r, "fix": f} for t, r, f in fixes]})


def audit_response(*verdicts: tuple[str, bool, str, str]) -> AIMessage:
    """QA's round-1 check of its own failing tests: (test, test_is_wrong, reason, corrected source).
    No verdicts means every failing test was judged correct."""
    return structured("TestAudit", {"verdicts": [
        {"test": t, "test_is_wrong": w, "reason": r, "corrected": c} for t, w, r, c in verdicts]})


def qa_agent(settings: Any, *responses: Any) -> Any:
    from agents.qa_agent import QAAgent

    return QAAgent(settings, model=ScriptedChatModel(responses=list(responses)))


# --- Coder self-reflection ------------------------------------------------------------

def critique_response(*items: tuple[str, str, str, str]) -> AIMessage:
    """items are (kind, location, concern, change); the default is one edge case about empty strings."""
    items = items or (("edge_case", "palindrome.py:is_pal", "empty string is not covered by a test",
                       "add a test that is_pal('') is True"),)
    return structured("Critique", {"items": [
        {"kind": k, "location": loc, "concern": c, "change": ch} for k, loc, c, ch in items]})


def task_scripts() -> tuple[list, list]:
    """(coder script, QA script) for one task that passes QA in round 1."""
    return happy() + [critique_response()] + revise_happy(), [qa_suite_response()]

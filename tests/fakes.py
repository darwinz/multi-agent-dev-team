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

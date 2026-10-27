"""Chat model construction and provider prerequisite checks."""

from __future__ import annotations

import json
import os
import urllib.request

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from agents.config import Settings


def make_model(settings: Settings) -> BaseChatModel:
    return init_chat_model(settings.model, max_tokens=settings.max_tokens)


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

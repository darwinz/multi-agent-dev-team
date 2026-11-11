"""On-disk response cache with TTL; one JSON file per key, written atomically."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, messages_from_dict, messages_to_dict


def _stable_default(o: object) -> str:
    """JSON fallback for `bound`: classes and callables by qualified name, never an address repr."""
    if isinstance(o, type) or callable(o):
        return f"{getattr(o, '__module__', '')}.{getattr(o, '__qualname__', type(o).__qualname__)}"
    raise TypeError(f"cache key: unserializable {type(o).__name__} in bound")


def _strip_volatile(d: dict) -> dict:
    """Drop per-response random ids and metadata so equal histories key equally across processes."""
    data = dict(d.get("data", {}))
    for f in ("id", "response_metadata", "usage_metadata", "tool_call_id"):
        data.pop(f, None)
    for f in ("tool_calls", "invalid_tool_calls"):
        if isinstance(data.get(f), list):
            data[f] = [{k: v for k, v in tc.items() if k != "id"} for tc in data[f]]
    if isinstance(data.get("content"), list):
        drop = {"tool_use": "id", "tool_result": "tool_use_id"}
        data["content"] = [
            {k: v for k, v in b.items() if k != drop[b["type"]]}
            if isinstance(b, dict) and b.get("type") in drop else b
            for b in data["content"]]
    ak = data.get("additional_kwargs")
    if isinstance(ak, dict) and isinstance(ak.get("tool_calls"), list):
        data["additional_kwargs"] = {**ak, "tool_calls": [
            {k: v for k, v in tc.items() if k != "id"} for tc in ak["tool_calls"]]}
    return {**d, "data": data}


class ResponseCache:
    """On-disk cache of model replies, keyed by a hash of the request; entries older than ttl_s count as expired."""

    def __init__(self, root: Path, ttl_s: float, clock: Callable[[], float] = time.time):
        """Cache under root with a TTL in seconds (0 disables it) and an injectable clock."""
        self.root = Path(root)
        self.ttl_s = ttl_s
        self.clock = clock

    @property
    def enabled(self) -> bool:
        """True when the TTL is positive."""
        return self.ttl_s > 0

    def key(self, model_id: str, messages: list[BaseMessage], bound: dict,
            stop: list[str] | None) -> str:
        """Sha256 of the model id, messages (volatile ids stripped), bound tools and stop sequences."""
        payload = {"model": model_id, "messages": [_strip_volatile(m) for m in messages_to_dict(messages)],
                   "bound": bound, "stop": stop}
        blob = json.dumps(payload, sort_keys=True, default=_stable_default)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> tuple[Literal["hit", "miss", "expired"], AIMessage | None]:
        """Look up a key: ('hit', msg), ('expired', None) or ('miss', None); disabled or unreadable entries miss."""
        if not self.enabled:
            return "miss", None
        try:
            data = json.loads(self._path(key).read_text(encoding="utf-8"))
            created_at = float(data["created_at"])
            msg = messages_from_dict([data["message"]])[0]
            if not isinstance(msg, AIMessage):
                return "miss", None
        except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError):
            return "miss", None
        if self.clock() - created_at > self.ttl_s:
            return "expired", None
        return "hit", msg

    def put(self, key: str, message: AIMessage) -> None:
        """Store a reply atomically (temp file then replace); no-op when disabled."""
        if not self.enabled:
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"created_at": self.clock(), "message": messages_to_dict([message])[0]}
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(record, f)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

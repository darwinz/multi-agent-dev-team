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


class ResponseCache:
    def __init__(self, root: Path, ttl_s: float, clock: Callable[[], float] = time.time):
        self.root = Path(root)
        self.ttl_s = ttl_s
        self.clock = clock

    @property
    def enabled(self) -> bool:
        return self.ttl_s > 0

    def key(self, model_id: str, messages: list[BaseMessage], bound: dict,
            stop: list[str] | None) -> str:
        payload = {"model": model_id, "messages": messages_to_dict(messages),
                   "bound": bound, "stop": stop}
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> tuple[Literal["hit", "miss", "expired"], AIMessage | None]:
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

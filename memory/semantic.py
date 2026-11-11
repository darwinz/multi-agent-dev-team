"""Chroma-backed semantic memory of past runs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import chromadb
from chromadb.errors import NotFoundError
from pydantic import BaseModel

from resilience.errors import PermanentError, TransientError, classify_chroma_error
from resilience.runtime import resilient_call

T = TypeVar("T")


class MemoryUnavailableError(TransientError):
    """The Chroma server could not be reached (after retries, or with the chroma breaker open)."""

    def __str__(self) -> str:
        """Show the message as given, without the wrapped-error prefix AgentError adds."""
        return str(self.args[0]) if self.args else super().__str__()


class MemoryHit(BaseModel):
    """One semantic-memory search result: entry id, stored text and cosine distance (lower is closer)."""

    id: str
    text: str
    distance: float


def memory_id(text: str, metadata: dict) -> str:
    """A deterministic id, so a retried (replayed) write upserts the same entry."""
    return hashlib.sha256((text + "\0" + json.dumps(metadata, sort_keys=True)).encode()).hexdigest()[:32]


def _chroma(fn: Callable[[], T], name: str) -> T:
    """Retry + the shared "chroma" breaker; a permanent failure re-raises Chroma's own exception."""
    try:
        return resilient_call(fn, breaker="chroma", classify=classify_chroma_error, name=f"chroma.{name}")[0]
    except PermanentError as e:
        if e.original is not None and e.__cause__ is e.original:  # wrapped by call_with_retry
            raise e.original from None
        raise


class SemanticMemory:
    """Chroma collection of past runs searched by cosine distance; calls go through retry and the chroma breaker."""

    def __init__(self, chroma_dir: Path, embedding_function: Any = None, collection: str = "coder_runs", *,
                 host: str | None = None, port: int = 8000) -> None:
        """Open the collection locally, or on a Chroma server if host is set (MemoryUnavailableError if down)."""
        self._name = collection
        self._embedding_function = embedding_function
        if host is None:
            self._client = _chroma(lambda: chromadb.PersistentClient(path=str(chroma_dir)), "client")
            self._collection = self._open()
            return
        try:
            # HttpClient contacts the server while it is constructed, so unreachability shows up here
            self._client = _chroma(lambda: chromadb.HttpClient(host=host, port=port), "client")
            self._collection = self._open()
        except Exception as e:
            raise MemoryUnavailableError(
                f"Chroma server unreachable at {host}:{port}: {e}",
                original=e.original if isinstance(e, TransientError) else e,
                attempts=e.attempts if isinstance(e, TransientError) else 1) from e

    def _open(self):
        kwargs: dict[str, Any] = {"configuration": {"hnsw": {"space": "cosine"}}}
        # Passing None explicitly would disable embedding; omit it to get Chroma's default local model
        if self._embedding_function is not None:
            kwargs["embedding_function"] = self._embedding_function
        return _chroma(lambda: self._client.get_or_create_collection(self._name, **kwargs), "open")

    def add(self, text: str, metadata: dict[str, str | int | float | bool], *, doc_id: str | None = None) -> str:
        # The id is fixed before the retry: every attempt upserts the same entry, so a replay is a no-op
        """Upsert a text with metadata under a deterministic id (or the given one) and return that id."""
        doc_id = doc_id or memory_id(text, metadata)
        _chroma(lambda: self._collection.upsert(ids=[doc_id], documents=[text], metadatas=[metadata]), "add")
        return doc_id

    def search(self, query: str, k: int, max_distance: float) -> list[MemoryHit]:
        """Up to k nearest entries within max_distance, closest first; empty when the collection is empty."""
        n = self.count()
        if n == 0:
            return []
        res = _chroma(lambda: self._collection.query(query_texts=[query], n_results=min(k, n)), "search")
        hits = [
            MemoryHit(id=i, text=doc, distance=dist)
            for i, doc, dist in zip(res["ids"][0], res["documents"][0], res["distances"][0])
            if dist <= max_distance
        ]
        return sorted(hits, key=lambda h: h.distance)

    def count(self) -> int:
        """Number of entries in the collection."""
        return _chroma(lambda: self._collection.count(), "count")

    def reopen(self) -> None:
        """Re-fetch the collection by name, e.g. after another process cleared (recreated) it."""
        self._collection = self._open()

    def clear(self) -> None:
        """Delete the collection and recreate it empty."""
        def delete() -> None:
            try:
                self._client.delete_collection(self._name)
            except NotFoundError:  # a replay after a delete that committed: already gone
                pass
        _chroma(delete, "clear")
        self._collection = self._open()

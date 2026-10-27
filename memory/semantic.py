"""Chroma-backed semantic memory of past runs."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import chromadb
from pydantic import BaseModel


class MemoryHit(BaseModel):
    id: str
    text: str
    distance: float


class SemanticMemory:
    def __init__(self, chroma_dir: Path, embedding_function: Any = None, collection: str = "coder_runs") -> None:
        self._client = chromadb.PersistentClient(path=str(chroma_dir))
        self._name = collection
        self._embedding_function = embedding_function
        self._collection = self._open()

    def _open(self):
        kwargs: dict[str, Any] = {"configuration": {"hnsw": {"space": "cosine"}}}
        # Passing None explicitly would disable embedding; omit it to get Chroma's default local model
        if self._embedding_function is not None:
            kwargs["embedding_function"] = self._embedding_function
        return self._client.get_or_create_collection(self._name, **kwargs)

    def add(self, text: str, metadata: dict[str, str | int | float | bool]) -> str:
        doc_id = uuid.uuid4().hex
        self._collection.add(ids=[doc_id], documents=[text], metadatas=[metadata])
        return doc_id

    def search(self, query: str, k: int, max_distance: float) -> list[MemoryHit]:
        n = self.count()
        if n == 0:
            return []
        res = self._collection.query(query_texts=[query], n_results=min(k, n))
        hits = [
            MemoryHit(id=i, text=doc, distance=dist)
            for i, doc, dist in zip(res["ids"][0], res["documents"][0], res["distances"][0])
            if dist <= max_distance
        ]
        return sorted(hits, key=lambda h: h.distance)

    def count(self) -> int:
        return self._collection.count()

    def reopen(self) -> None:
        """Re-fetch the collection by name, e.g. after another process cleared (recreated) it."""
        self._collection = self._open()

    def clear(self) -> None:
        self._client.delete_collection(self._name)
        self._collection = self._open()

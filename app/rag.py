from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from math import sqrt
import os
from pathlib import Path
import re
import tempfile
from threading import Lock
from typing import Any, List, Optional, Protocol
from uuid import uuid4

@dataclass
class RagChunk:
    doc_id: str
    title: str
    chunk_id: int
    text: str
    embedding: list[float]
    knowledge_base_id: str = "default"
    doc_hash: str = ""
    chunk_hash: str = ""
    source: str = "manual"
    tags: list[str] | None = None
    language: str = "unknown"
    created_at: str = ""


@dataclass
class RagMatch:
    doc_id: str
    title: str
    chunk_id: int
    text: str
    score: float
    knowledge_base_id: str = "default"
    doc_hash: str = ""
    chunk_hash: str = ""
    source: str = "manual"
    tags: list[str] | None = None
    language: str = "unknown"
    created_at: str = ""


def normalize_tags(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        # `|` was used by older Chroma records; keep it readable while using
        # comma-separated values for new records.
        return [item.strip() for item in re.split(r"[,|]", value) if item.strip()]
    return []


def normalize_metadata(metadata: Optional[dict[str, object]]) -> dict[str, object]:
    knowledge_base_id = str((metadata or {}).get("knowledge_base_id") or "default").strip() or "default"
    doc_hash = str((metadata or {}).get("doc_hash") or "").strip()
    source = str((metadata or {}).get("source") or "manual").strip() or "manual"
    language = str((metadata or {}).get("language") or "unknown").strip() or "unknown"
    tags = normalize_tags((metadata or {}).get("tags"))
    created_at = str((metadata or {}).get("created_at") or "").strip()
    if not created_at:
        created_at = datetime.now(timezone.utc).isoformat()
    return {
        "knowledge_base_id": knowledge_base_id,
        "doc_hash": doc_hash,
        "source": source,
        "language": language,
        "tags": tags,
        "created_at": created_at,
    }


def chunk_matches_filters(chunk: RagChunk, filters: Optional[dict[str, object]]) -> bool:
    if not filters:
        return True

    doc_id_filter = str(filters.get("doc_id") or "").strip()
    if doc_id_filter and chunk.doc_id != doc_id_filter:
        return False

    kb_filter = str(filters.get("knowledge_base_id") or "").strip()
    if kb_filter and chunk.knowledge_base_id != kb_filter:
        return False

    doc_hash_filter = str(filters.get("doc_hash") or "").strip()
    if doc_hash_filter and chunk.doc_hash != doc_hash_filter:
        return False

    chunk_hash_filter = str(filters.get("chunk_hash") or "").strip()
    if chunk_hash_filter and chunk.chunk_hash != chunk_hash_filter:
        return False

    source_filter = str(filters.get("source") or "").strip().lower()
    if source_filter and chunk.source.lower() != source_filter:
        return False

    language_filter = str(filters.get("language") or "").strip().lower()
    if language_filter and chunk.language.lower() != language_filter:
        return False

    tag_filters = {item.lower() for item in normalize_tags(filters.get("tags"))}
    if tag_filters:
        chunk_tags = {item.lower() for item in (chunk.tags or [])}
        if chunk_tags.isdisjoint(tag_filters):
            return False

    return True


class RagStore(Protocol):
    def add_document(
        self,
        title: str,
        chunk_texts: list[str],
        chunk_embeddings: list[list[float]],
        metadata: Optional[dict[str, object]] = None,
    ) -> tuple[str, int]:
        ...

    def list_documents(self, filters: Optional[dict[str, object]] = None) -> list[dict[str, object]]:
        ...

    def delete_document(self, doc_id: str, filters: Optional[dict[str, object]] = None) -> bool:
        ...

    def clear_all(self, filters: Optional[dict[str, object]] = None) -> int:
        ...

    def search(self, query_embedding: list[float], top_k: int, filters: Optional[dict[str, object]] = None) -> list[RagMatch]:
        ...

    def list_chunks(self, filters: Optional[dict[str, object]] = None) -> list[RagChunk]:
        ...


def split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    cleaned = text.strip()
    if not cleaned:
        return []

    if chunk_size <= 0:
        return [cleaned]

    step = max(1, chunk_size - max(0, overlap))
    chunks: List[str] = []
    start = 0
    while start < len(cleaned):
        chunks.append(cleaned[start : start + chunk_size].strip())
        start += step

    return [chunk for chunk in chunks if chunk]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0

    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sqrt(sum(a * a for a in left))
    right_norm = sqrt(sum(b * b for b in right))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot / (left_norm * right_norm)


class LocalJsonRagStore:
    def __init__(self, store_path: str) -> None:
        self._chunks: list[RagChunk] = []
        self._lock = Lock()
        self._store_path = Path(store_path)
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        if not self._store_path.exists():
            return

        raw_text = self._store_path.read_text(encoding="utf-8").strip()
        if not raw_text:
            return

        payload = json.loads(raw_text)
        raw_chunks = payload.get("chunks", [])
        if not isinstance(raw_chunks, list):
            return

        self._chunks = [
            RagChunk(
                doc_id=str(item["doc_id"]),
                title=str(item["title"]),
                chunk_id=int(item["chunk_id"]),
                text=str(item["text"]),
                embedding=[float(value) for value in item["embedding"]],
                knowledge_base_id=str(item.get("knowledge_base_id") or "default"),
                doc_hash=str(item.get("doc_hash") or ""),
                chunk_hash=str(item.get("chunk_hash") or ""),
                source=str(item.get("source") or "manual"),
                tags=normalize_tags(item.get("tags")),
                language=str(item.get("language") or "unknown"),
                created_at=str(item.get("created_at") or ""),
            )
            for item in raw_chunks
            if isinstance(item, dict)
            and "doc_id" in item
            and "title" in item
            and "chunk_id" in item
            and "text" in item
            and isinstance(item.get("embedding"), list)
        ]

    def _save_to_disk(self) -> None:
        payload = {
            "chunks": [
                {
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "embedding": chunk.embedding,
                    "knowledge_base_id": chunk.knowledge_base_id,
                    "doc_hash": chunk.doc_hash,
                    "chunk_hash": chunk.chunk_hash,
                    "source": chunk.source,
                    "tags": chunk.tags or [],
                    "language": chunk.language,
                    "created_at": chunk.created_at,
                }
                for chunk in self._chunks
            ]
        }
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        fd, temp_path = tempfile.mkstemp(
            dir=self._store_path.parent,
            prefix=f".{self._store_path.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self._store_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def add_document(
        self,
        title: str,
        chunk_texts: list[str],
        chunk_embeddings: list[list[float]],
        metadata: Optional[dict[str, object]] = None,
    ) -> tuple[str, int]:
        if len(chunk_texts) != len(chunk_embeddings):
            raise ValueError("chunk text count must match embedding count")

        raw_metadata = metadata or {}
        normalized = normalize_metadata(metadata)
        doc_id = str(uuid4())
        chunks = [
            RagChunk(
                doc_id=doc_id,
                title=title,
                chunk_id=index,
                text=chunk_text,
                embedding=chunk_embeddings[index],
                knowledge_base_id=str(normalized["knowledge_base_id"]),
                doc_hash=str(normalized["doc_hash"]),
                chunk_hash=str(raw_metadata.get("chunk_hashes", [""] * len(chunk_texts))[index]),
                source=str(normalized["source"]),
                tags=list(normalized["tags"]),
                language=str(normalized["language"]),
                created_at=str(normalized["created_at"]),
            )
            for index, chunk_text in enumerate(chunk_texts)
        ]

        with self._lock:
            self._chunks.extend(chunks)
            self._save_to_disk()

        return doc_id, len(chunks)

    def list_documents(self, filters: Optional[dict[str, object]] = None) -> list[dict[str, object]]:
        summary: dict[str, dict[str, object]] = {}
        with self._lock:
            for chunk in self._chunks:
                if not chunk_matches_filters(chunk, filters):
                    continue
                if chunk.doc_id not in summary:
                    summary[chunk.doc_id] = {
                        "doc_id": chunk.doc_id,
                        "title": chunk.title,
                        "chunks": 0,
                        "knowledge_base_id": chunk.knowledge_base_id,
                        "doc_hash": chunk.doc_hash,
                        "source": chunk.source,
                        "tags": chunk.tags or [],
                        "language": chunk.language,
                        "created_at": chunk.created_at,
                    }
                summary[chunk.doc_id]["chunks"] = int(summary[chunk.doc_id]["chunks"]) + 1

        return list(summary.values())

    def delete_document(self, doc_id: str, filters: Optional[dict[str, object]] = None) -> bool:
        with self._lock:
            original = len(self._chunks)
            matches_document = any(
                chunk.doc_id == doc_id and chunk_matches_filters(chunk, filters)
                for chunk in self._chunks
            )
            self._chunks = [
                chunk
                for chunk in self._chunks
                if not (matches_document and chunk.doc_id == doc_id)
            ]
            changed = len(self._chunks) != original
            if changed:
                self._save_to_disk()
            return changed

    def clear_all(self, filters: Optional[dict[str, object]] = None) -> int:
        with self._lock:
            if not filters:
                doc_count = len({chunk.doc_id for chunk in self._chunks})
                self._chunks = []
                self._save_to_disk()
                return doc_count

            matching_doc_ids = {
                chunk.doc_id for chunk in self._chunks if chunk_matches_filters(chunk, filters)
            }
            doc_count = len(matching_doc_ids)
            self._chunks = [chunk for chunk in self._chunks if chunk.doc_id not in matching_doc_ids]
            self._save_to_disk()
            return doc_count

    def search(self, query_embedding: list[float], top_k: int, filters: Optional[dict[str, object]] = None) -> list[RagMatch]:
        with self._lock:
            scored = [
                RagMatch(
                    doc_id=chunk.doc_id,
                    title=chunk.title,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=cosine_similarity(query_embedding, chunk.embedding),
                    knowledge_base_id=chunk.knowledge_base_id,
                    doc_hash=chunk.doc_hash,
                    chunk_hash=chunk.chunk_hash,
                    source=chunk.source,
                    tags=chunk.tags or [],
                    language=chunk.language,
                    created_at=chunk.created_at,
                )
                for chunk in self._chunks
                if chunk_matches_filters(chunk, filters)
            ]

        scored.sort(key=lambda item: item.score, reverse=True)
        return [match for match in scored[:top_k] if match.score > 0]

    def list_chunks(self, filters: Optional[dict[str, object]] = None) -> list[RagChunk]:
        with self._lock:
            return [chunk for chunk in self._chunks if chunk_matches_filters(chunk, filters)]


class ChromaRagStore:
    def __init__(self, persist_directory: str, collection_name: str) -> None:
        import chromadb

        path = Path(persist_directory)
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def add_document(
        self,
        title: str,
        chunk_texts: list[str],
        chunk_embeddings: list[list[float]],
        metadata: Optional[dict[str, object]] = None,
    ) -> tuple[str, int]:
        if len(chunk_texts) != len(chunk_embeddings):
            raise ValueError("chunk text count must match embedding count")

        raw_metadata = metadata or {}
        normalized = normalize_metadata(metadata)
        doc_id = str(uuid4())
        ids = [f"{doc_id}:{index}" for index in range(len(chunk_texts))]
        metadatas = [
            {
                "doc_id": doc_id,
                "title": title,
                "chunk_id": index,
                "knowledge_base_id": str(normalized["knowledge_base_id"]),
                "doc_hash": str(normalized["doc_hash"]),
                "chunk_hash": str(raw_metadata.get("chunk_hashes", [""] * len(chunk_texts))[index]),
                "source": str(normalized["source"]),
                "language": str(normalized["language"]),
                "tags": ",".join(list(normalized["tags"])),
                "created_at": str(normalized["created_at"]),
            }
            for index in range(len(chunk_texts))
        ]

        self._collection.add(
            ids=ids,
            documents=chunk_texts,
            embeddings=chunk_embeddings,
            metadatas=metadatas,
        )

        return doc_id, len(chunk_texts)

    def list_documents(self, filters: Optional[dict[str, object]] = None) -> list[dict[str, object]]:
        result = self._collection.get(include=["metadatas"])
        metadatas = result.get("metadatas") or []

        summary: dict[str, dict[str, object]] = {}
        for metadata in metadatas:
            if not isinstance(metadata, dict):
                continue
            doc_id = str(metadata.get("doc_id", ""))
            if not doc_id:
                continue

            chunk = RagChunk(
                doc_id=doc_id,
                title=str(metadata.get("title", "Untitled")),
                chunk_id=int(metadata.get("chunk_id", 0)),
                text="",
                embedding=[],
                knowledge_base_id=str(metadata.get("knowledge_base_id", "default")),
                doc_hash=str(metadata.get("doc_hash", "")),
                chunk_hash=str(metadata.get("chunk_hash", "")),
                source=str(metadata.get("source", "manual")),
                tags=normalize_tags(str(metadata.get("tags", ""))),
                language=str(metadata.get("language", "unknown")),
                created_at=str(metadata.get("created_at", "")),
            )
            if not chunk_matches_filters(chunk, filters):
                continue

            if doc_id not in summary:
                summary[doc_id] = {
                    "doc_id": doc_id,
                    "title": str(metadata.get("title", "Untitled")),
                    "chunks": 0,
                    "knowledge_base_id": chunk.knowledge_base_id,
                    "doc_hash": chunk.doc_hash,
                    "source": chunk.source,
                    "tags": chunk.tags or [],
                    "language": chunk.language,
                    "created_at": chunk.created_at,
                }
            summary[doc_id]["chunks"] = int(summary[doc_id]["chunks"]) + 1

        return list(summary.values())

    def delete_document(self, doc_id: str, filters: Optional[dict[str, object]] = None) -> bool:
        result = self._collection.get(include=["metadatas"])
        matching_doc_ids = {
            str(metadata.get("doc_id", ""))
            for metadata in result.get("metadatas") or []
            if isinstance(metadata, dict)
            and str(metadata.get("doc_id", "")) == doc_id
            and chunk_matches_filters(self._chunk_from_metadata(metadata), filters)
        }
        ids = [
            item_id
            for item_id, metadata in zip(result.get("ids") or [], result.get("metadatas") or [])
            if isinstance(metadata, dict)
            and str(metadata.get("doc_id", "")) in matching_doc_ids
        ]
        if not ids:
            return False

        self._collection.delete(ids=ids)
        return True

    def clear_all(self, filters: Optional[dict[str, object]] = None) -> int:
        result = self._collection.get(include=["metadatas"])
        matching_doc_ids = {
            self._chunk_from_metadata(metadata).doc_id
            for metadata in result.get("metadatas") or []
            if isinstance(metadata, dict) and chunk_matches_filters(self._chunk_from_metadata(metadata), filters)
        }
        if not matching_doc_ids:
            return 0

        ids = [
            item_id
            for item_id, metadata in zip(result.get("ids") or [], result.get("metadatas") or [])
            if isinstance(metadata, dict) and str(metadata.get("doc_id", "")) in matching_doc_ids
        ]
        if ids:
            self._collection.delete(ids=ids)
        return len(matching_doc_ids)

    def search(self, query_embedding: list[float], top_k: int, filters: Optional[dict[str, object]] = None) -> list[RagMatch]:
        collection_count = self._collection.count()
        if collection_count <= 0:
            return []

        # When filters are present, retrieve all eligible candidates before
        # applying tag filtering in Python; a fixed 8x window can otherwise
        # discard valid matches that happen to rank below unrelated chunks.
        candidate_count = collection_count if filters else min(max(top_k * 8, top_k), collection_count)
        query_kwargs: dict[str, object] = {
            "query_embeddings": [query_embedding],
            "n_results": candidate_count,
            "include": ["documents", "metadatas", "distances"],
        }
        where = self._build_where(filters)
        if where:
            query_kwargs["where"] = where

        result = self._collection.query(**query_kwargs)

        documents_group = result.get("documents") or [[]]
        metadatas_group = result.get("metadatas") or [[]]
        distances_group = result.get("distances") or [[]]

        documents = documents_group[0] if documents_group else []
        metadatas = metadatas_group[0] if metadatas_group else []
        distances = distances_group[0] if distances_group else []

        matches: list[RagMatch] = []
        for index, document in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) and isinstance(metadatas[index], dict) else {}
            distance = float(distances[index]) if index < len(distances) else 0.0
            score = 1.0 / (1.0 + max(distance, 0.0))
            match = RagMatch(
                doc_id=str(metadata.get("doc_id", "")),
                title=str(metadata.get("title", "Untitled")),
                chunk_id=int(metadata.get("chunk_id", index)),
                text=str(document),
                score=score,
                knowledge_base_id=str(metadata.get("knowledge_base_id", "default")),
                doc_hash=str(metadata.get("doc_hash", "")),
                chunk_hash=str(metadata.get("chunk_hash", "")),
                source=str(metadata.get("source", "manual")),
                tags=normalize_tags(str(metadata.get("tags", ""))),
                language=str(metadata.get("language", "unknown")),
                created_at=str(metadata.get("created_at", "")),
            )
            if chunk_matches_filters(
                RagChunk(
                    doc_id=match.doc_id,
                    title=match.title,
                    chunk_id=match.chunk_id,
                    text=match.text,
                    embedding=[],
                    knowledge_base_id=match.knowledge_base_id,
                    doc_hash=match.doc_hash,
                    chunk_hash=match.chunk_hash,
                    source=match.source,
                    tags=match.tags,
                    language=match.language,
                    created_at=match.created_at,
                ),
                filters,
            ):
                matches.append(match)

        return matches[:top_k]

    @staticmethod
    def _chunk_from_metadata(metadata: dict[str, object]) -> RagChunk:
        return RagChunk(
            doc_id=str(metadata.get("doc_id", "")),
            title=str(metadata.get("title", "Untitled")),
            chunk_id=int(metadata.get("chunk_id", 0)),
            text="",
            embedding=[],
            knowledge_base_id=str(metadata.get("knowledge_base_id", "default")),
            doc_hash=str(metadata.get("doc_hash", "")),
            chunk_hash=str(metadata.get("chunk_hash", "")),
            source=str(metadata.get("source", "manual")),
            tags=normalize_tags(metadata.get("tags", "")),
            language=str(metadata.get("language", "unknown")),
            created_at=str(metadata.get("created_at", "")),
        )

    @staticmethod
    def _build_where(filters: Optional[dict[str, object]]) -> dict[str, object] | None:
        if not filters:
            return None

        conditions: list[dict[str, object]] = []
        # Keep case-insensitive source/language matching in Python. Applying
        # those fields to Chroma's case-sensitive `where` would reject legacy
        # records whose metadata uses different casing.
        for field in ("knowledge_base_id", "doc_hash"):
            value = str(filters.get(field) or "").strip()
            if value:
                conditions.append({field: value})
        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def list_chunks(self, filters: Optional[dict[str, object]] = None) -> list[RagChunk]:
        result = self._collection.get(include=["documents", "metadatas"])
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []

        chunks: list[RagChunk] = []
        for index, document in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) and isinstance(metadatas[index], dict) else {}
            chunks.append(
                RagChunk(
                    doc_id=str(metadata.get("doc_id", "")),
                    title=str(metadata.get("title", "Untitled")),
                    chunk_id=int(metadata.get("chunk_id", index)),
                    text=str(document),
                    embedding=[],
                    knowledge_base_id=str(metadata.get("knowledge_base_id", "default")),
                    doc_hash=str(metadata.get("doc_hash", "")),
                    chunk_hash=str(metadata.get("chunk_hash", "")),
                    source=str(metadata.get("source", "manual")),
                    tags=normalize_tags(str(metadata.get("tags", ""))),
                    language=str(metadata.get("language", "unknown")),
                    created_at=str(metadata.get("created_at", "")),
                )
            )

        return [chunk for chunk in chunks if chunk_matches_filters(chunk, filters)]


def create_rag_store(
    backend: str,
    rag_store_path: str,
    chroma_persist_directory: str,
    chroma_collection_name: str,
) -> RagStore:
    normalized = backend.strip().lower()
    if normalized == "chroma":
        return ChromaRagStore(
            persist_directory=chroma_persist_directory,
            collection_name=chroma_collection_name,
        )
    if normalized == "local_json":
        return LocalJsonRagStore(store_path=rag_store_path)

    raise ValueError("Invalid RAG backend. Supported values: 'chroma', 'local_json'")

# Backward-compatible alias for existing imports.
InMemoryRagStore = LocalJsonRagStore

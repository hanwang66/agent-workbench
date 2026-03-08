from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import logging
from math import log
import re
from threading import Lock
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .ollama_client import OllamaClient
from .rag import RagChunk, RagMatch, create_rag_store, split_text


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
LOGGER = logging.getLogger(__name__)


@dataclass
class TranslationTurn:
    source_text: str
    translated_text: str


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def build_translation_prompt(
    text: str,
    source_lang: str,
    target_lang: str,
    style: str,
    domain: str,
    glossary: Dict[str, str],
    history: List[TranslationTurn],
    rag_matches: List[RagMatch],
) -> str:
    glossary_block = ""
    if glossary:
        glossary_lines = [f"- {k} => {v}" for k, v in glossary.items()]
        glossary_block = "\nPreferred glossary (must follow when applicable):\n" + "\n".join(glossary_lines)

    history_block = ""
    if history:
        history_lines = [
            f"Turn {index + 1}\nSource: {turn.source_text}\nTranslation: {turn.translated_text}"
            for index, turn in enumerate(history)
        ]
        history_block = "\nRecent translation memory:\n" + "\n\n".join(history_lines)

    rag_block = ""
    if rag_matches:
        rag_lines = [
            f"- [{match.title}#{match.chunk_id}] {match.text}"
            for match in rag_matches
        ]
        rag_block = "\nRetrieved knowledge (use when relevant and accurate):\n" + "\n".join(rag_lines)

    return (
        "You are a professional translation agent.\n"
        f"Translate the following text from {source_lang} to {target_lang}.\n"
        f"Style: {style}.\n"
        f"Domain: {domain}.\n"
        "Rules:\n"
        "1. Return only the translated text.\n"
        "2. Keep punctuation and meaning accurate.\n"
        "3. Do not add explanations.\n\n"
        f"{glossary_block}\n"
        f"{history_block}\n"
        f"{rag_block}\n"
        f"Text:\n{text}"
    )


class TranslationService:
    def __init__(
        self,
        client: OllamaClient,
        max_session_turns: int = 6,
        rag_embed_model: str = "nomic-embed-text",
        rag_chunk_size: int = 500,
        rag_chunk_overlap: int = 80,
        rag_backend: str = "chroma",
        rag_store_path: str = "data/rag_store.json",
        chroma_persist_directory: str = "data/chroma",
        chroma_collection_name: str = "translation_rag",
        retrieval_mode: str = "hybrid",
        hybrid_vector_weight: float = 0.6,
        hybrid_bm25_weight: float = 0.4,
        hybrid_rrf_k: int = 60,
        hybrid_candidate_multiplier: int = 4,
        function_call_max_rounds: int = 4,
        function_call_max_tool_calls: int = 4,
        function_call_allowed_tools: str = "get_rag_context",
    ) -> None:
        self._client = client
        self._max_session_turns = max_session_turns
        self._rag_embed_model = rag_embed_model
        self._rag_chunk_size = rag_chunk_size
        self._rag_chunk_overlap = rag_chunk_overlap
        self._retrieval_mode = retrieval_mode
        self._hybrid_vector_weight = hybrid_vector_weight
        self._hybrid_bm25_weight = hybrid_bm25_weight
        self._hybrid_rrf_k = max(1, hybrid_rrf_k)
        self._hybrid_candidate_multiplier = max(1, hybrid_candidate_multiplier)
        self._function_call_max_rounds = max(1, function_call_max_rounds)
        self._function_call_max_tool_calls = max(1, function_call_max_tool_calls)
        self._function_call_allowed_tools = {
            item.strip() for item in function_call_allowed_tools.split(",") if item.strip()
        }
        if not self._function_call_allowed_tools:
            self._function_call_allowed_tools = {"get_rag_context"}
        self._session_memory: Dict[str, List[TranslationTurn]] = defaultdict(list)
        self._lock = Lock()
        self._rag_store = create_rag_store(
            backend=rag_backend,
            rag_store_path=rag_store_path,
            chroma_persist_directory=chroma_persist_directory,
            chroma_collection_name=chroma_collection_name,
        )

    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        style: str,
        domain: str,
        glossary: Dict[str, str],
        use_rag: bool,
        use_function_calling: bool,
        rag_top_k: int,
        retrieval_mode: Optional[str] = None,
        rag_filter: Optional[dict[str, object]] = None,
        knowledge_base_id: str = "default",
        session_id: Optional[str] = None,
    ) -> tuple[str, str, int, bool, list[str], list[dict[str, object]]]:
        active_session_id = session_id or str(uuid4())
        history = self.get_session_memory(active_session_id)
        base_filters = {"knowledge_base_id": knowledge_base_id, **(rag_filter or {})}
        tool_traces: list[dict[str, object]] = []

        if use_function_calling:
            translated, rag_matches, tool_traces = await self._translate_with_function_calling(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                style=style,
                domain=domain,
                glossary=glossary,
                history=history,
                use_rag=use_rag,
                rag_top_k=rag_top_k,
                retrieval_mode=retrieval_mode,
                filters=base_filters,
            )
        else:
            rag_matches = (
                await self.retrieve_knowledge(
                    text=text,
                    top_k=rag_top_k,
                    retrieval_mode=retrieval_mode,
                    filters=base_filters,
                )
                if use_rag
                else []
            )
            prompt = build_translation_prompt(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                style=style,
                domain=domain,
                glossary=glossary,
                history=history,
                rag_matches=rag_matches,
            )
            translated = await self._client.generate(prompt)
        memory_turns = self.append_turn(active_session_id, text, translated)
        rag_chunks = [f"{item.title}#{item.chunk_id}" for item in rag_matches]
        return translated, active_session_id, memory_turns, bool(rag_matches), rag_chunks, tool_traces

    def _build_fc_system_prompt(
        self,
        source_lang: str,
        target_lang: str,
        style: str,
        domain: str,
        glossary: Dict[str, str],
        history: List[TranslationTurn],
        use_rag: bool,
    ) -> str:
        glossary_block = ""
        if glossary:
            glossary_lines = [f"- {k} => {v}" for k, v in glossary.items()]
            glossary_block = "\nGlossary:\n" + "\n".join(glossary_lines)

        history_block = ""
        if history:
            history_lines = [
                f"Turn {index + 1}: {turn.source_text} => {turn.translated_text}"
                for index, turn in enumerate(history)
            ]
            history_block = "\nRecent memory:\n" + "\n".join(history_lines)

        rag_instruction = "You may call get_rag_context when you need external terminology/context." if use_rag else "Do not call tools."

        return (
            "You are a professional translation agent.\n"
            f"Translate from {source_lang} to {target_lang}.\n"
            f"Style: {style}. Domain: {domain}.\n"
            "Rules: return only translated text, preserve meaning and punctuation.\n"
            f"{rag_instruction}"
            f"{glossary_block}"
            f"{history_block}"
        )

    @staticmethod
    def _parse_tool_arguments(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                return {}
        return {}

    async def _translate_with_function_calling(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        style: str,
        domain: str,
        glossary: Dict[str, str],
        history: List[TranslationTurn],
        use_rag: bool,
        rag_top_k: int,
        retrieval_mode: Optional[str],
        filters: dict[str, object],
    ) -> tuple[str, list[RagMatch], list[dict[str, object]]]:
        system_prompt = self._build_fc_system_prompt(
            source_lang=source_lang,
            target_lang=target_lang,
            style=style,
            domain=domain,
            glossary=glossary,
            history=history,
            use_rag=use_rag,
        )
        user_prompt = f"Source text:\n{text}"
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        tools: list[dict[str, Any]] = []
        if use_rag:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_rag_context",
                        "description": "Retrieve related translation context from knowledge base",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "top_k": {"type": "integer"},
                                "retrieval_mode": {"type": "string"},
                            },
                            "required": ["query"],
                        },
                    },
                }
            ]

        tool_traces: list[dict[str, object]] = []
        tool_calls_count = 0
        last_content = ""
        rag_matches: list[RagMatch] = []

        for round_index in range(1, self._function_call_max_rounds + 1):
            reply = await self._client.chat(messages=messages, tools=tools if use_rag else None)
            content = str(reply.get("content") or "").strip()
            if content:
                last_content = content
            tool_calls = reply.get("tool_calls")

            if not (use_rag and isinstance(tool_calls, list) and tool_calls):
                if last_content:
                    return last_content, rag_matches, tool_traces
                continue

            messages.append({"role": "assistant", "content": str(reply.get("content") or ""), "tool_calls": tool_calls})

            for call in tool_calls:
                fn_payload = call.get("function") if isinstance(call, dict) and isinstance(call.get("function"), dict) else {}
                fn_name = str(fn_payload.get("name") or "")
                args = self._parse_tool_arguments(fn_payload.get("arguments"))

                if tool_calls_count >= self._function_call_max_tool_calls:
                    trace = {
                        "round_index": round_index,
                        "tool_name": fn_name or "<unknown>",
                        "status": "blocked",
                        "detail": "tool call limit reached",
                        "result_count": 0,
                    }
                    tool_traces.append(trace)
                    LOGGER.warning("tool_call_trace=%s", trace)
                    messages.append({"role": "tool", "name": fn_name or "unknown", "content": json.dumps({"error": trace["detail"]})})
                    continue

                if fn_name not in self._function_call_allowed_tools:
                    trace = {
                        "round_index": round_index,
                        "tool_name": fn_name or "<unknown>",
                        "status": "blocked",
                        "detail": "tool is not in whitelist",
                        "result_count": 0,
                    }
                    tool_traces.append(trace)
                    LOGGER.warning("tool_call_trace=%s", trace)
                    messages.append({"role": "tool", "name": fn_name or "unknown", "content": json.dumps({"error": trace["detail"]})})
                    continue

                tool_calls_count += 1
                if fn_name == "get_rag_context":
                    try:
                        query = str(args.get("query") or text)
                        tool_top_k = int(args.get("top_k") or rag_top_k)
                        tool_mode_raw = args.get("retrieval_mode")
                        tool_mode = str(tool_mode_raw) if isinstance(tool_mode_raw, str) else retrieval_mode

                        rag_matches = await self.retrieve_knowledge(
                            text=query,
                            top_k=max(1, min(tool_top_k, 8)),
                            retrieval_mode=tool_mode,
                            filters=filters,
                        )
                        tool_result = [
                            {
                                "doc_id": match.doc_id,
                                "title": match.title,
                                "chunk_id": match.chunk_id,
                                "text": match.text,
                                "score": match.score,
                            }
                            for match in rag_matches
                        ]
                        trace = {
                            "round_index": round_index,
                            "tool_name": fn_name,
                            "status": "executed",
                            "detail": "ok",
                            "result_count": len(tool_result),
                        }
                        tool_traces.append(trace)
                        LOGGER.info("tool_call_trace=%s", trace)
                        messages.append(
                            {
                                "role": "tool",
                                "name": fn_name,
                                "content": json.dumps(tool_result, ensure_ascii=False),
                            }
                        )
                    except Exception as exc:
                        trace = {
                            "round_index": round_index,
                            "tool_name": fn_name,
                            "status": "error",
                            "detail": str(exc),
                            "result_count": 0,
                        }
                        tool_traces.append(trace)
                        LOGGER.warning("tool_call_trace=%s", trace)
                        messages.append({"role": "tool", "name": fn_name, "content": json.dumps({"error": str(exc)})})

        if last_content:
            return last_content, rag_matches, tool_traces
        raise ValueError("Function-calling translation produced empty output")

    async def retrieve_knowledge(
        self,
        text: str,
        top_k: int,
        retrieval_mode: Optional[str] = None,
        filters: Optional[dict[str, object]] = None,
    ) -> list[RagMatch]:
        documents = self.list_rag_documents(filters=filters)
        if not documents:
            return []

        mode = (retrieval_mode or self._retrieval_mode).strip().lower()
        if mode not in {"vector", "bm25", "hybrid"}:
            mode = "hybrid"

        if mode == "vector":
            return await self._vector_retrieve(text=text, top_k=top_k, filters=filters)
        if mode == "bm25":
            return self._bm25_retrieve(text=text, top_k=top_k, filters=filters)
        return await self._hybrid_retrieve(text=text, top_k=top_k, filters=filters)

    async def _vector_retrieve(self, text: str, top_k: int, filters: Optional[dict[str, object]] = None) -> list[RagMatch]:
        embedding = await self._client.embed(text=text, embed_model=self._rag_embed_model)
        return self._rag_store.search(query_embedding=embedding, top_k=top_k, filters=filters)

    def _bm25_retrieve(self, text: str, top_k: int, filters: Optional[dict[str, object]] = None) -> list[RagMatch]:
        chunks = self._rag_store.list_chunks(filters=filters)
        if not chunks:
            return []

        query_tokens = tokenize(text)
        if not query_tokens:
            return []

        tokenized_docs = [tokenize(chunk.text) for chunk in chunks]
        doc_lengths = [len(tokens) for tokens in tokenized_docs]
        avg_doc_len = sum(doc_lengths) / max(1, len(doc_lengths))
        k1 = 1.5
        b = 0.75

        # Document frequency for query terms only.
        df: Dict[str, int] = {token: 0 for token in query_tokens}
        for tokens in tokenized_docs:
            unique = set(tokens)
            for token in df:
                if token in unique:
                    df[token] += 1

        n_docs = len(chunks)
        scored: list[RagMatch] = []
        for index, chunk in enumerate(chunks):
            tokens = tokenized_docs[index]
            if not tokens:
                continue

            term_freq: Dict[str, int] = {}
            for token in tokens:
                term_freq[token] = term_freq.get(token, 0) + 1

            score = 0.0
            for token in query_tokens:
                freq = term_freq.get(token, 0)
                if freq == 0:
                    continue
                term_df = df.get(token, 0)
                idf = log(1 + (n_docs - term_df + 0.5) / (term_df + 0.5))
                denom = freq + k1 * (1 - b + b * (doc_lengths[index] / max(avg_doc_len, 1e-9)))
                score += idf * (freq * (k1 + 1)) / max(denom, 1e-9)

            if score > 0:
                scored.append(
                    RagMatch(
                        doc_id=chunk.doc_id,
                        title=chunk.title,
                        chunk_id=chunk.chunk_id,
                        text=chunk.text,
                        score=score,
                        knowledge_base_id=chunk.knowledge_base_id,
                        source=chunk.source,
                        tags=chunk.tags,
                        language=chunk.language,
                        created_at=chunk.created_at,
                    )
                )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    async def _hybrid_retrieve(self, text: str, top_k: int, filters: Optional[dict[str, object]] = None) -> list[RagMatch]:
        candidate_k = max(top_k, top_k * self._hybrid_candidate_multiplier)
        vector_matches = await self._vector_retrieve(text=text, top_k=candidate_k, filters=filters)
        bm25_matches = self._bm25_retrieve(text=text, top_k=candidate_k, filters=filters)

        fused: dict[str, RagMatch] = {}
        scores: dict[str, float] = {}

        for rank, match in enumerate(vector_matches):
            key = f"{match.doc_id}:{match.chunk_id}"
            fused[key] = match
            scores[key] = scores.get(key, 0.0) + self._hybrid_vector_weight / (self._hybrid_rrf_k + rank + 1)

        for rank, match in enumerate(bm25_matches):
            key = f"{match.doc_id}:{match.chunk_id}"
            if key not in fused:
                fused[key] = match
            scores[key] = scores.get(key, 0.0) + self._hybrid_bm25_weight / (self._hybrid_rrf_k + rank + 1)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        output: list[RagMatch] = []
        for key, score in ranked[:top_k]:
            match = fused[key]
            output.append(
                RagMatch(
                    doc_id=match.doc_id,
                    title=match.title,
                    chunk_id=match.chunk_id,
                    text=match.text,
                    score=score,
                )
            )

        return output

    async def ingest_rag_document(
        self,
        title: str,
        text: str,
        metadata: Optional[dict[str, object]] = None,
    ) -> tuple[str, int]:
        chunks = split_text(text=text, chunk_size=self._rag_chunk_size, overlap=self._rag_chunk_overlap)
        if not chunks:
            raise ValueError("Document text is empty after preprocessing")

        input_metadata = metadata or {}
        kb_id = str(input_metadata.get("knowledge_base_id") or "default").strip() or "default"
        doc_hash = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

        existing_docs = self._rag_store.list_documents(filters={"knowledge_base_id": kb_id, "doc_hash": doc_hash})
        if existing_docs:
            raise ValueError("Duplicate document detected in this knowledge base")

        existing_chunks = self._rag_store.list_chunks(filters={"knowledge_base_id": kb_id})
        existing_chunk_hashes = {chunk.chunk_hash for chunk in existing_chunks if chunk.chunk_hash}

        unique_chunks: list[str] = []
        unique_chunk_hashes: list[str] = []
        seen_in_request: set[str] = set()
        for chunk in chunks:
            chunk_hash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
            if chunk_hash in seen_in_request or chunk_hash in existing_chunk_hashes:
                continue
            seen_in_request.add(chunk_hash)
            unique_chunks.append(chunk)
            unique_chunk_hashes.append(chunk_hash)

        if not unique_chunks:
            raise ValueError("All chunks are duplicates in this knowledge base")

        embeddings: list[list[float]] = []
        for chunk in unique_chunks:
            embeddings.append(await self._client.embed(text=chunk, embed_model=self._rag_embed_model))

        normalized_metadata = {
            **input_metadata,
            "knowledge_base_id": kb_id,
            "doc_hash": doc_hash,
            "chunk_hashes": unique_chunk_hashes,
        }

        return self._rag_store.add_document(
            title=title,
            chunk_texts=unique_chunks,
            chunk_embeddings=embeddings,
            metadata=normalized_metadata,
        )

    def list_rag_documents(self, filters: Optional[dict[str, object]] = None) -> list[dict[str, object]]:
        return self._rag_store.list_documents(filters=filters)

    def delete_rag_document(self, doc_id: str, filters: Optional[dict[str, object]] = None) -> bool:
        return self._rag_store.delete_document(doc_id, filters=filters)

    def clear_rag_documents(self, filters: Optional[dict[str, object]] = None) -> int:
        return self._rag_store.clear_all(filters=filters)

    def append_turn(self, session_id: str, source_text: str, translated_text: str) -> int:
        with self._lock:
            turns = self._session_memory[session_id]
            turns.append(TranslationTurn(source_text=source_text, translated_text=translated_text))
            if len(turns) > self._max_session_turns:
                self._session_memory[session_id] = turns[-self._max_session_turns :]
            return len(self._session_memory[session_id])

    def get_session_memory(self, session_id: str) -> List[TranslationTurn]:
        with self._lock:
            return list(self._session_memory.get(session_id, []))

    def get_session_turn_count(self, session_id: str) -> int:
        with self._lock:
            return len(self._session_memory.get(session_id, []))

    def clear_session(self, session_id: str) -> bool:
        with self._lock:
            existed = session_id in self._session_memory
            self._session_memory.pop(session_id, None)
            return existed

    async def check_readiness(self) -> tuple[bool, bool, list[str]]:
        available_models = await self._client.list_models()
        model_available = self._client.model in available_models
        embed_model_available = self._rag_embed_model in available_models
        return model_available, embed_model_available, available_models

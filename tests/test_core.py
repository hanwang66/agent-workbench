from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import ValidationError

from app.rag import RagChunk, chunk_matches_filters, normalize_tags, split_text
from app.schemas import TranslateRequest
from app.service import TranslationTurn, TranslationService, build_translation_prompt
from app.metrics import MetricsCollector


class FakeClient:
    model = "fake-model"

    async def list_models(self) -> list[str]:
        return ["fake-model", "nomic-embed-text"]

    async def embed(self, text: str, embed_model: str) -> list[float]:
        return [float(len(text)), 1.0]

    async def generate(self, prompt: str) -> str:
        return "translated"

    async def chat(self, messages: list[dict[str, object]], tools: list[dict[str, object]] | None = None) -> dict[str, object]:
        return {"content": "translated"}


class FakeFunctionCallingClient(FakeClient):
    def __init__(self) -> None:
        self._turn = 0

    async def chat(self, messages: list[dict[str, object]], tools: list[dict[str, object]] | None = None) -> dict[str, object]:
        self._turn += 1
        if self._turn == 1:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_rag_context",
                            "arguments": '{"query":"model context","top_k":1,"retrieval_mode":"bm25"}',
                        }
                    }
                ],
            }
        return {"content": "function calling translation"}


class FakeBlockedToolClient(FakeClient):
    def __init__(self) -> None:
        self._turn = 0

    async def chat(self, messages: list[dict[str, object]], tools: list[dict[str, object]] | None = None) -> dict[str, object]:
        self._turn += 1
        if self._turn == 1:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "web_search",
                            "arguments": '{"query":"x"}',
                        }
                    }
                ],
            }
        return {"content": "translation after blocked tool"}


class FakeLimitToolClient(FakeClient):
    def __init__(self) -> None:
        self._turn = 0

    async def chat(self, messages: list[dict[str, object]], tools: list[dict[str, object]] | None = None) -> dict[str, object]:
        self._turn += 1
        if self._turn == 1:
            return {
                "content": "",
                "tool_calls": [
                    {"function": {"name": "get_rag_context", "arguments": '{"query":"a","top_k":1}'}},
                    {"function": {"name": "get_rag_context", "arguments": '{"query":"b","top_k":1}'}},
                ],
            }
        return {"content": "translation after limit"}


class FakeRagStore:
    def __init__(self, chunks: list[RagChunk]) -> None:
        self._chunks = chunks

    def list_chunks(self, filters: dict[str, object] | None = None) -> list[RagChunk]:
        return [chunk for chunk in self._chunks if chunk_matches_filters(chunk, filters)]

    def list_documents(self, filters: dict[str, object] | None = None) -> list[dict[str, object]]:
        summary: dict[str, dict[str, object]] = {}
        for chunk in self.list_chunks(filters=filters):
            if chunk.doc_id not in summary:
                summary[chunk.doc_id] = {"doc_id": chunk.doc_id, "title": chunk.title, "chunks": 0}
            summary[chunk.doc_id]["chunks"] = int(summary[chunk.doc_id]["chunks"]) + 1
        return list(summary.values())


class CoreTests(unittest.TestCase):
    def test_metrics_render_prometheus(self) -> None:
        metrics = MetricsCollector()
        metrics.observe_http("get", "/health", 200, 0.125)
        metrics.observe_http("get", "/health", 200, 0.075)

        output = metrics.render_prometheus()
        self.assertIn("agent_workbench_http_requests_total", output)
        self.assertIn('method="GET",path="/health",status="200"} 2', output)
        self.assertIn('agent_workbench_http_request_duration_seconds_count{method="GET",path="/health"} 2', output)

    def test_split_text_with_overlap(self) -> None:
        text = "abcdefghij"
        chunks = split_text(text=text, chunk_size=4, overlap=1)
        self.assertEqual(chunks, ["abcd", "defg", "ghij", "j"])

    def test_prompt_builder_includes_controls(self) -> None:
        prompt = build_translation_prompt(
            text="你好",
            source_lang="Chinese",
            target_lang="English",
            style="formal",
            domain="technical",
            glossary={"模型": "model"},
            history=[TranslationTurn(source_text="你好", translated_text="hello")],
            rag_matches=[],
        )
        self.assertIn("Style: formal", prompt)
        self.assertIn("Domain: technical", prompt)
        self.assertIn("模型 => model", prompt)
        self.assertIn("Recent translation memory", prompt)

    def test_translate_request_retrieval_mode_validation(self) -> None:
        request = TranslateRequest(
            text="hi",
            retrieval_mode="hybrid",
            knowledge_base_id="team-a",
            rag_filter={"knowledge_base_id": "team-a", "source": "manual", "language": "en", "tags": ["ai"]},
        )
        self.assertEqual(request.retrieval_mode, "hybrid")
        self.assertEqual(request.knowledge_base_id, "team-a")
        self.assertEqual(request.rag_filter.source, "manual")
        self.assertEqual(request.rag_filter.language, "en")

        with self.assertRaises(ValidationError):
            TranslateRequest(text="hi", retrieval_mode="invalid")

    def test_rag_filter_cannot_override_knowledge_base(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = TranslationService(
                client=FakeClient(),
                rag_backend="local_json",
                rag_store_path=f"{tmp}/rag_store.json",
            )
            service._rag_store = FakeRagStore(
                chunks=[
                    RagChunk(
                        doc_id="team-a-doc",
                        title="Team A",
                        chunk_id=0,
                        text="model context for team a",
                        embedding=[],
                        knowledge_base_id="team-a",
                    ),
                    RagChunk(
                        doc_id="team-b-doc",
                        title="Team B",
                        chunk_id=0,
                        text="model context for team b",
                        embedding=[],
                        knowledge_base_id="team-b",
                    ),
                ]
            )

            _, _, _, rag_used, rag_chunks, _ = asyncio.run(
                service.translate(
                    text="model context",
                    source_lang="English",
                    target_lang="Chinese",
                    style="neutral",
                    domain="general",
                    glossary={},
                    use_rag=True,
                    use_function_calling=False,
                    rag_top_k=2,
                    retrieval_mode="bm25",
                    rag_filter={"knowledge_base_id": "team-b"},
                    knowledge_base_id="team-a",
                )
            )

            self.assertTrue(rag_used)
            self.assertEqual(rag_chunks, ["Team A#0"])

    def test_legacy_pipe_separated_tags_are_read(self) -> None:
        self.assertEqual(normalize_tags("terminology|nlp"), ["terminology", "nlp"])

    def test_session_count_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = TranslationService(
                client=FakeClient(),
                rag_backend="local_json",
                rag_store_path=f"{tmp}/rag_store.json",
                max_sessions=2,
                session_ttl_seconds=0,
            )

            service.append_turn("session-a", "a", "A")
            service.append_turn("session-b", "b", "B")
            service.append_turn("session-c", "c", "C")

            self.assertEqual(service.get_session_turn_count("session-a"), 0)
            self.assertEqual(service.get_session_turn_count("session-b"), 1)
            self.assertEqual(service.get_session_turn_count("session-c"), 1)

    def test_bm25_retrieve_matches_best_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = TranslationService(
                client=FakeClient(),
                rag_backend="local_json",
                rag_store_path=f"{tmp}/rag_store.json",
            )

            service._rag_store = FakeRagStore(
                chunks=[
                    RagChunk(doc_id="d1", title="Doc1", chunk_id=0, text="transformer model context window", embedding=[]),
                    RagChunk(doc_id="d2", title="Doc2", chunk_id=0, text="weather forecast and rainfall", embedding=[]),
                ]
            )

            matches = service._bm25_retrieve(text="model context", top_k=2)
            self.assertGreaterEqual(len(matches), 1)
            self.assertEqual(matches[0].doc_id, "d1")

    def test_bm25_retrieve_respects_language_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = TranslationService(
                client=FakeClient(),
                rag_backend="local_json",
                rag_store_path=f"{tmp}/rag_store.json",
            )

            service._rag_store = FakeRagStore(
                chunks=[
                    RagChunk(
                        doc_id="d1",
                        title="Doc1",
                        chunk_id=0,
                        text="model context retrieval",
                        embedding=[],
                        knowledge_base_id="team-a",
                        language="en",
                        source="manual",
                        tags=["ai"],
                    ),
                    RagChunk(
                        doc_id="d2",
                        title="Doc2",
                        chunk_id=0,
                        text="model context retrieval",
                        embedding=[],
                        knowledge_base_id="team-b",
                        language="zh",
                        source="manual",
                        tags=["ai"],
                    ),
                ]
            )

            matches = service._bm25_retrieve(
                text="model context",
                top_k=5,
                filters={"knowledge_base_id": "team-a", "language": "en"},
            )
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0].doc_id, "d1")

    def test_duplicate_document_hash_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = TranslationService(
                client=FakeClient(),
                rag_backend="local_json",
                rag_store_path=f"{tmp}/rag_store.json",
                rag_chunk_size=5,
                rag_chunk_overlap=0,
            )

            asyncio.run(
                service.ingest_rag_document(
                    title="Doc1",
                    text="aaaaabbbbb",
                    metadata={"knowledge_base_id": "team-a"},
                )
            )

            with self.assertRaises(ValueError):
                asyncio.run(
                    service.ingest_rag_document(
                        title="Doc1 duplicate",
                        text="aaaaabbbbb",
                        metadata={"knowledge_base_id": "team-a"},
                    )
                )

    def test_duplicate_chunks_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = TranslationService(
                client=FakeClient(),
                rag_backend="local_json",
                rag_store_path=f"{tmp}/rag_store.json",
                rag_chunk_size=5,
                rag_chunk_overlap=0,
            )

            first_doc_id, first_chunks = asyncio.run(
                service.ingest_rag_document(
                    title="Doc1",
                    text="aaaaabbbbb",
                    metadata={"knowledge_base_id": "team-a"},
                )
            )
            self.assertTrue(first_doc_id)
            self.assertEqual(first_chunks, 2)

            second_doc_id, second_chunks = asyncio.run(
                service.ingest_rag_document(
                    title="Doc2",
                    text="bbbbbccccc",
                    metadata={"knowledge_base_id": "team-a"},
                )
            )
            self.assertTrue(second_doc_id)
            self.assertEqual(second_chunks, 1)

    def test_function_calling_translation_with_rag_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = TranslationService(
                client=FakeFunctionCallingClient(),
                rag_backend="local_json",
                rag_store_path=f"{tmp}/rag_store.json",
            )

            service._rag_store = FakeRagStore(
                chunks=[
                    RagChunk(
                        doc_id="d1",
                        title="Doc1",
                        chunk_id=0,
                        text="model context translation note",
                        embedding=[],
                        knowledge_base_id="team-a",
                        language="en",
                    )
                ]
            )

            translated, _, _, rag_used, rag_chunks, tool_traces = asyncio.run(
                service.translate(
                    text="模型上下文",
                    source_lang="Chinese",
                    target_lang="English",
                    style="neutral",
                    domain="technical",
                    glossary={},
                    use_rag=True,
                    use_function_calling=True,
                    rag_top_k=2,
                    retrieval_mode="bm25",
                    rag_filter=None,
                    knowledge_base_id="team-a",
                )
            )

            self.assertEqual(translated, "function calling translation")
            self.assertTrue(rag_used)
            self.assertEqual(len(rag_chunks), 1)
            self.assertEqual(len(tool_traces), 1)
            self.assertEqual(tool_traces[0]["status"], "executed")

    def test_function_calling_blocks_non_whitelisted_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = TranslationService(
                client=FakeBlockedToolClient(),
                rag_backend="local_json",
                rag_store_path=f"{tmp}/rag_store.json",
            )

            translated, _, _, _, _, tool_traces = asyncio.run(
                service.translate(
                    text="测试",
                    source_lang="Chinese",
                    target_lang="English",
                    style="neutral",
                    domain="general",
                    glossary={},
                    use_rag=True,
                    use_function_calling=True,
                    rag_top_k=1,
                    retrieval_mode="bm25",
                    rag_filter=None,
                    knowledge_base_id="default",
                )
            )

            self.assertEqual(translated, "translation after blocked tool")
            self.assertEqual(len(tool_traces), 1)
            self.assertEqual(tool_traces[0]["status"], "blocked")

    def test_function_calling_enforces_tool_call_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = TranslationService(
                client=FakeLimitToolClient(),
                rag_backend="local_json",
                rag_store_path=f"{tmp}/rag_store.json",
                function_call_max_tool_calls=1,
            )

            service._rag_store = FakeRagStore(
                chunks=[RagChunk(doc_id="d1", title="Doc1", chunk_id=0, text="a", embedding=[])]
            )

            translated, _, _, _, _, tool_traces = asyncio.run(
                service.translate(
                    text="测试",
                    source_lang="Chinese",
                    target_lang="English",
                    style="neutral",
                    domain="general",
                    glossary={},
                    use_rag=True,
                    use_function_calling=True,
                    rag_top_k=1,
                    retrieval_mode="bm25",
                    rag_filter=None,
                    knowledge_base_id="default",
                )
            )

            self.assertEqual(translated, "translation after limit")
            self.assertEqual(len(tool_traces), 2)
            self.assertEqual(tool_traces[0]["status"], "executed")
            self.assertEqual(tool_traces[1]["status"], "blocked")


if __name__ == "__main__":
    unittest.main()

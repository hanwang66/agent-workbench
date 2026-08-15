from __future__ import annotations

from typing import Any

from ..schemas import TranslateRequest
from ..service import TranslationService
from .base import AgentContext, AgentResult, AgentTask


class TranslationAgent:
    name = "translation"
    description = "Translate text with session memory, glossary controls, and optional RAG."
    capabilities = ("translation", "rag", "glossary", "translation_memory")

    def __init__(self, service: TranslationService, model: str) -> None:
        self._service = service
        self._model = model

    async def run(self, task: AgentTask, context: AgentContext) -> AgentResult:
        payload: dict[str, Any] = dict(task.parameters)
        translation_text = task.input_text
        previous_outputs = [result.output for result in context.previous_results if result.output.strip()]
        if previous_outputs and payload.get("include_previous_results", True) is True:
            translation_text = (
                f"{task.input_text}\n\nPrevious worker output:\n"
                + "\n\n".join(previous_outputs)
            )
        payload["text"] = translation_text
        payload["knowledge_base_id"] = task.knowledge_base_id
        if task.session_id is not None:
            payload["session_id"] = task.session_id

        request = TranslateRequest.model_validate(payload)
        translated, session_id, memory_turns, rag_used, rag_chunks, tool_traces = await self._service.translate(
            text=request.text,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            style=request.style,
            domain=request.domain,
            glossary=request.glossary,
            use_rag=request.use_rag,
            use_function_calling=request.use_function_calling,
            rag_top_k=request.rag_top_k,
            retrieval_mode=request.retrieval_mode,
            rag_filter=request.rag_filter.model_dump(exclude_none=True) if request.rag_filter else None,
            knowledge_base_id=request.knowledge_base_id,
            session_id=request.session_id,
        )

        return AgentResult(
            agent_name=self.name,
            status="completed",
            output=translated,
            metadata={
                "model": self._model,
                "source_lang": request.source_lang,
                "target_lang": request.target_lang,
                "session_id": session_id,
                "memory_turns": memory_turns,
                "rag_used": rag_used,
                "rag_chunks": rag_chunks,
                "previous_results_used": bool(previous_outputs and payload.get("include_previous_results", True) is True),
            },
            tool_traces=tool_traces,
        )

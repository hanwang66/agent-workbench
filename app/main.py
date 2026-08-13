from __future__ import annotations

from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import httpx
from pathlib import Path

from .config import settings
from .errors import ApiError
from .ollama_client import OllamaClient
from .schemas import (
    ClearSessionResponse,
    ErrorResponse,
    HealthResponse,
    ReadinessResponse,
    RagClearAllResponse,
    RagDeleteResponse,
    RagDocumentInfo,
    RagIngestRequest,
    RagIngestResponse,
    RagSearchFilter,
    SessionInfoResponse,
    TranslateRequest,
    TranslateResponse,
)
from .service import TranslationService

app = FastAPI(title="Ollama Translation Agent", version="0.1.0")

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

ollama_client = OllamaClient(
    base_url=settings.ollama_base_url,
    model=settings.ollama_model,
    temperature=settings.ollama_temperature,
    timeout_seconds=settings.ollama_timeout_seconds,
    max_retries=settings.ollama_max_retries,
    retry_backoff_seconds=settings.ollama_retry_backoff_seconds,
)
translation_service = TranslationService(
    client=ollama_client,
    max_session_turns=settings.max_session_turns,
    max_sessions=settings.max_sessions,
    session_ttl_seconds=settings.session_ttl_seconds,
    rag_embed_model=settings.ollama_embed_model,
    rag_chunk_size=settings.rag_chunk_size,
    rag_chunk_overlap=settings.rag_chunk_overlap,
    rag_backend=settings.rag_backend,
    rag_store_path=settings.rag_store_path,
    chroma_persist_directory=settings.chroma_persist_directory,
    chroma_collection_name=settings.chroma_collection_name,
    retrieval_mode=settings.retrieval_mode,
    hybrid_vector_weight=settings.hybrid_vector_weight,
    hybrid_bm25_weight=settings.hybrid_bm25_weight,
    hybrid_rrf_k=settings.hybrid_rrf_k,
    hybrid_candidate_multiplier=settings.hybrid_candidate_multiplier,
    function_call_max_rounds=settings.function_call_max_rounds,
    function_call_max_tool_calls=settings.function_call_max_tool_calls,
    function_call_allowed_tools=settings.function_call_allowed_tools,
)


def decode_uploaded_text(raw_bytes: bytes) -> str:
    # Try common encodings for Chinese/English text files.
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Unsupported file encoding; use UTF-8 or GB18030 text files")


def normalize_extensions(raw_extensions: str) -> set[str]:
    return {item.strip().lower() for item in raw_extensions.split(",") if item.strip()}


def validate_upload(file: UploadFile, content: bytes) -> None:
    filename = (file.filename or "").strip()
    extension = Path(filename).suffix.lower()
    allowed_extensions = normalize_extensions(settings.upload_allowed_extensions)
    content_type = (file.content_type or "").lower()

    if len(content) > settings.max_upload_size_bytes:
        raise ApiError(
            status_code=413,
            code="FILE_TOO_LARGE",
            message=f"Upload exceeds limit ({settings.max_upload_size_bytes} bytes)",
        )

    if allowed_extensions and extension not in allowed_extensions:
        raise ApiError(
            status_code=400,
            code="FILE_TYPE_UNSUPPORTED",
            message=f"Unsupported file extension '{extension or '<none>'}'",
        )

    if content_type and not (content_type.startswith("text/") or content_type in {"application/octet-stream"}):
        raise ApiError(
            status_code=400,
            code="FILE_TYPE_UNSUPPORTED",
            message=f"Unsupported content type '{content_type}'",
        )


async def read_upload_with_limit(file: UploadFile) -> bytes:
    """Read an upload incrementally so oversized files are rejected early."""
    limit = settings.max_upload_size_bytes
    parts: list[bytes] = []
    total = 0

    while True:
        # Read at most one byte beyond the configured limit so the check does
        # not require buffering an arbitrarily large multipart upload.
        read_size = min(64 * 1024, max(1, limit - total + 1))
        chunk = await file.read(read_size)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise ApiError(
                status_code=413,
                code="FILE_TOO_LARGE",
                message=f"Upload exceeds limit ({limit} bytes)",
            )
        parts.append(chunk)

    return b"".join(parts)


@app.exception_handler(ApiError)
async def api_error_handler(_: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error={"code": exc.code, "message": exc.message}).model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(error={"code": "REQUEST_VALIDATION_ERROR", "message": str(exc)}).model_dump(),
    )


@app.get("/")
async def home() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        ollama_base_url=settings.ollama_base_url,
        model=settings.ollama_model,
    )


@app.get("/health/ready", response_model=ReadinessResponse, responses={503: {"model": ErrorResponse}})
async def readiness() -> ReadinessResponse:
    try:
        model_available, embed_model_available, available_models = await translation_service.check_readiness()
    except httpx.TimeoutException as exc:
        raise ApiError(status_code=504, code="OLLAMA_TIMEOUT", message=f"Readiness check timed out: {exc}") from exc
    except httpx.HTTPError as exc:
        raise ApiError(status_code=502, code="OLLAMA_UNREACHABLE", message=f"Cannot reach Ollama: {exc}") from exc
    except Exception as exc:
        raise ApiError(status_code=500, code="READINESS_CHECK_FAILED", message=f"Readiness check failed: {exc}") from exc

    if not model_available or not embed_model_available:
        raise ApiError(
            status_code=503,
            code="MODEL_NOT_READY",
            message=(
                f"Configured models are not fully available. model={settings.ollama_model}, "
                f"embed_model={settings.ollama_embed_model}"
            ),
        )

    return ReadinessResponse(
        status="ready",
        model=settings.ollama_model,
        embed_model=settings.ollama_embed_model,
        model_available=model_available,
        embed_model_available=embed_model_available,
        available_models=available_models,
    )


@app.post("/translate", response_model=TranslateResponse)
async def translate(request: TranslateRequest) -> TranslateResponse:
    try:
        translated, session_id, memory_turns, rag_used, rag_chunks, tool_traces = await translation_service.translate(
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
    except httpx.TimeoutException as exc:
        raise ApiError(status_code=504, code="OLLAMA_TIMEOUT", message=f"Ollama request timed out: {exc}") from exc
    except httpx.HTTPStatusError as exc:
        detail = f"Ollama returned HTTP {exc.response.status_code}: {exc.response.text}"
        raise ApiError(status_code=502, code="OLLAMA_HTTP_ERROR", message=detail) from exc
    except httpx.HTTPError as exc:
        raise ApiError(status_code=502, code="OLLAMA_UNREACHABLE", message=f"Cannot reach Ollama: {exc}") from exc
    except Exception as exc:
        raise ApiError(status_code=500, code="TRANSLATION_FAILED", message=f"Translation failed: {exc}") from exc

    return TranslateResponse(
        translated_text=translated,
        model=settings.ollama_model,
        source_lang=request.source_lang,
        target_lang=request.target_lang,
        session_id=session_id,
        memory_turns=memory_turns,
        rag_used=rag_used,
        rag_chunks=rag_chunks,
        tool_traces=tool_traces,
    )


@app.get("/sessions/{session_id}", response_model=SessionInfoResponse)
async def get_session(session_id: str) -> SessionInfoResponse:
    return SessionInfoResponse(
        session_id=session_id,
        memory_turns=translation_service.get_session_turn_count(session_id),
    )


@app.delete("/sessions/{session_id}", response_model=ClearSessionResponse)
async def clear_session(session_id: str) -> ClearSessionResponse:
    cleared = translation_service.clear_session(session_id)
    return ClearSessionResponse(session_id=session_id, cleared=cleared)


@app.post("/rag/documents", response_model=RagIngestResponse)
async def ingest_rag_document(request: RagIngestRequest) -> RagIngestResponse:
    try:
        metadata = {
            "knowledge_base_id": request.knowledge_base_id,
            "source": request.source,
            "tags": request.tags,
            "language": request.language,
            "created_at": request.created_at,
        }
        doc_id, chunks = await translation_service.ingest_rag_document(
            title=request.title,
            text=request.text,
            metadata=metadata,
        )
        docs = translation_service.list_rag_documents(filters={"doc_id": doc_id})
        doc_info = docs[0] if docs else {}
    except httpx.TimeoutException as exc:
        raise ApiError(status_code=504, code="EMBEDDING_TIMEOUT", message=f"Embedding request timed out: {exc}") from exc
    except httpx.HTTPStatusError as exc:
        detail = f"Ollama returned HTTP {exc.response.status_code}: {exc.response.text}"
        raise ApiError(status_code=502, code="EMBEDDING_HTTP_ERROR", message=detail) from exc
    except httpx.HTTPError as exc:
        raise ApiError(status_code=502, code="EMBEDDING_UNREACHABLE", message=f"Cannot reach Ollama embeddings: {exc}") from exc
    except ValueError as exc:
        raise ApiError(status_code=400, code="RAG_INGEST_INVALID_INPUT", message=str(exc)) from exc
    except Exception as exc:
        raise ApiError(status_code=500, code="RAG_INGEST_FAILED", message=f"RAG ingest failed: {exc}") from exc

    return RagIngestResponse(
        doc_id=doc_id,
        title=request.title,
        chunks=chunks,
        knowledge_base_id=str(doc_info.get("knowledge_base_id", request.knowledge_base_id)),
        source=str(doc_info.get("source", request.source)),
        tags=[str(tag) for tag in doc_info.get("tags", request.tags)],
        language=str(doc_info.get("language", request.language)),
        created_at=str(doc_info.get("created_at", request.created_at or "")),
    )


@app.post("/rag/documents/upload", response_model=RagIngestResponse)
async def upload_rag_document(
    file: UploadFile = File(...),
    title: str = Form(default=""),
    knowledge_base_id: str = Form(default="default"),
    source: str = Form(default="upload"),
    tags: str = Form(default=""),
    language: str = Form(default="unknown"),
    created_at: str = Form(default=""),
) -> RagIngestResponse:
    try:
        content = await read_upload_with_limit(file)
        validate_upload(file=file, content=content)
        try:
            text = decode_uploaded_text(content).strip()
        except ValueError as exc:
            raise ApiError(status_code=400, code="FILE_ENCODING_UNSUPPORTED", message=str(exc)) from exc
        if not text:
            raise ApiError(status_code=400, code="FILE_EMPTY", message="Uploaded file is empty")

        resolved_title = title.strip() or (file.filename or "Untitled")
        metadata = {
            "knowledge_base_id": knowledge_base_id,
            "source": source,
            "tags": [item.strip() for item in tags.split(",") if item.strip()],
            "language": language,
            "created_at": created_at.strip() or None,
        }
        doc_id, chunks = await translation_service.ingest_rag_document(
            title=resolved_title,
            text=text,
            metadata=metadata,
        )
        docs = translation_service.list_rag_documents(filters={"doc_id": doc_id})
        doc_info = docs[0] if docs else {}
    except httpx.TimeoutException as exc:
        raise ApiError(status_code=504, code="EMBEDDING_TIMEOUT", message=f"Embedding request timed out: {exc}") from exc
    except httpx.HTTPStatusError as exc:
        detail = f"Ollama returned HTTP {exc.response.status_code}: {exc.response.text}"
        raise ApiError(status_code=502, code="EMBEDDING_HTTP_ERROR", message=detail) from exc
    except httpx.HTTPError as exc:
        raise ApiError(status_code=502, code="EMBEDDING_UNREACHABLE", message=f"Cannot reach Ollama embeddings: {exc}") from exc
    except ApiError:
        raise
    except ValueError as exc:
        raise ApiError(status_code=400, code="RAG_INGEST_INVALID_INPUT", message=str(exc)) from exc
    except Exception as exc:
        raise ApiError(status_code=500, code="RAG_UPLOAD_FAILED", message=f"RAG upload failed: {exc}") from exc

    return RagIngestResponse(
        doc_id=doc_id,
        title=resolved_title,
        chunks=chunks,
        knowledge_base_id=str(doc_info.get("knowledge_base_id", knowledge_base_id)),
        source=str(doc_info.get("source", source)),
        tags=[str(tag) for tag in doc_info.get("tags", metadata["tags"])],
        language=str(doc_info.get("language", language)),
        created_at=str(doc_info.get("created_at", created_at)),
    )


@app.get("/rag/documents", response_model=list[RagDocumentInfo])
async def list_rag_documents(
    knowledge_base_id: str = Query(default="default"),
    source: str | None = Query(default=None),
    language: str | None = Query(default=None),
    tag: str | None = Query(default=None),
) -> list[RagDocumentInfo]:
    filters = RagSearchFilter(
        knowledge_base_id=knowledge_base_id,
        source=source,
        language=language,
        tags=[tag] if tag else [],
    )
    docs = translation_service.list_rag_documents(filters=filters.model_dump(exclude_none=True))
    return [
        RagDocumentInfo(
            doc_id=str(doc["doc_id"]),
            title=str(doc["title"]),
            chunks=int(doc["chunks"]),
            knowledge_base_id=str(doc.get("knowledge_base_id", "default")),
            source=str(doc.get("source", "manual")),
            tags=[str(tag) for tag in doc.get("tags", [])],
            language=str(doc.get("language", "unknown")),
            created_at=str(doc.get("created_at", "")),
        )
        for doc in docs
    ]


@app.delete("/rag/documents/{doc_id}", response_model=RagDeleteResponse)
async def delete_rag_document(doc_id: str, knowledge_base_id: str = Query(default="default")) -> RagDeleteResponse:
    deleted = translation_service.delete_rag_document(doc_id, filters={"knowledge_base_id": knowledge_base_id})
    return RagDeleteResponse(doc_id=doc_id, deleted=deleted)


@app.delete("/rag/documents", response_model=RagClearAllResponse)
async def clear_rag_documents(knowledge_base_id: str = Query(default="default")) -> RagClearAllResponse:
    deleted_documents = translation_service.clear_rag_documents(filters={"knowledge_base_id": knowledge_base_id})
    return RagClearAllResponse(deleted_documents=deleted_documents)

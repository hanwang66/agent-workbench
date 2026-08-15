from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


class ToolTrace(BaseModel):
    round_index: int
    tool_name: str
    status: Literal["executed", "blocked", "error"]
    detail: str
    result_count: int = 0


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Source text to translate")
    source_lang: str = Field(default="Chinese", description="Source language")
    target_lang: str = Field(default="English", description="Target language")
    session_id: Optional[str] = Field(default=None, description="Session ID for context memory")
    style: str = Field(default="neutral", description="Tone style like neutral, formal, casual")
    domain: str = Field(default="general", description="Domain like general, legal, medical, technical")
    glossary: Dict[str, str] = Field(default_factory=dict, description="Fixed term mapping for translation")
    use_rag: bool = Field(default=False, description="Whether to enable retrieval-augmented translation")
    use_function_calling: bool = Field(default=False, description="Enable tool/function-calling mode via Ollama chat API")
    rag_top_k: int = Field(default=3, ge=1, le=8, description="How many knowledge chunks to retrieve")
    retrieval_mode: Optional[Literal["vector", "bm25", "hybrid"]] = Field(
        default=None,
        description="RAG retrieval mode override: vector | bm25 | hybrid",
    )
    rag_filter: Optional["RagSearchFilter"] = Field(
        default=None,
        description="RAG metadata filter (source/language/tags)",
    )
    knowledge_base_id: str = Field(default="default", description="Knowledge base/project isolation ID")


class TranslateResponse(BaseModel):
    translated_text: str
    model: str
    source_lang: str
    target_lang: str
    session_id: str
    memory_turns: int
    rag_used: bool
    rag_chunks: list[str]
    tool_traces: list[ToolTrace] = Field(default_factory=list)


class AgentRunRequest(BaseModel):
    task: str = Field(..., min_length=1, description="User task for the orchestrator")
    agent_type: Optional[str] = Field(default=None, description="Optional worker name, e.g. translation or coding")
    session_id: Optional[str] = Field(default=None, description="Session ID shared with the selected worker")
    knowledge_base_id: str = Field(default="default", description="Knowledge base/project isolation ID")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Worker-specific parameters")


class AgentStepResponse(BaseModel):
    agent_name: str
    status: Literal["completed", "failed", "waiting_approval"]
    output: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    tool_traces: list[dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None


class AgentRunResponse(BaseModel):
    task_id: str
    agent_type: str
    status: Literal["completed", "failed", "waiting_approval"]
    output: str
    routing_reason: str
    steps: list[AgentStepResponse] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class AgentTaskStatusResponse(BaseModel):
    task_id: str
    status: Literal["queued", "running", "completed", "failed", "waiting_approval"]
    agent_type: str
    routing_reason: str
    current_agent: Optional[str] = None
    output: str = ""
    steps: list[AgentStepResponse] = Field(default_factory=list)
    error: Optional[str] = None
    created_at: str
    updated_at: str


class AgentInfoResponse(BaseModel):
    name: str
    description: str
    capabilities: list[str]


class HealthResponse(BaseModel):
    status: str
    ollama_base_url: str
    model: str


class ReadinessResponse(BaseModel):
    status: Literal["ready", "degraded"]
    model: str
    embed_model: str
    model_available: bool
    embed_model_available: bool
    available_models: list[str]


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class SessionInfoResponse(BaseModel):
    session_id: str
    memory_turns: int


class ClearSessionResponse(BaseModel):
    session_id: str
    cleared: bool


class RagIngestRequest(BaseModel):
    title: str = Field(default="Untitled", description="Document title")
    text: str = Field(..., min_length=1, description="Document text content")
    knowledge_base_id: str = Field(default="default", description="Knowledge base/project isolation ID")
    source: str = Field(default="manual", description="Document source, e.g. manual/upload/api")
    tags: list[str] = Field(default_factory=list, description="Document tags")
    language: str = Field(default="unknown", description="Document language")
    created_at: Optional[str] = Field(default=None, description="RFC3339 timestamp; auto-filled when omitted")


class RagIngestResponse(BaseModel):
    doc_id: str
    title: str
    chunks: int
    knowledge_base_id: str
    source: str
    tags: list[str]
    language: str
    created_at: str


class RagSearchFilter(BaseModel):
    knowledge_base_id: Optional[str] = None
    source: Optional[str] = None
    language: Optional[str] = None
    tags: list[str] = Field(default_factory=list)


class RagDocumentInfo(BaseModel):
    doc_id: str
    title: str
    chunks: int
    knowledge_base_id: str
    source: str
    tags: list[str]
    language: str
    created_at: str


class RagDeleteResponse(BaseModel):
    doc_id: str
    deleted: bool


class RagClearAllResponse(BaseModel):
    deleted_documents: int

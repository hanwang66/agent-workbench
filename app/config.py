from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:3b"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_temperature: float = 0.1
    ollama_timeout_seconds: int = 60
    ollama_max_retries: int = 2
    ollama_retry_backoff_seconds: float = 0.5
    max_session_turns: int = 6
    max_sessions: int = 10000
    session_ttl_seconds: int = 3600
    rag_chunk_size: int = 500
    rag_chunk_overlap: int = 80
    rag_backend: str = "chroma"
    rag_store_path: str = "data/rag_store.json"
    chroma_persist_directory: str = "data/chroma"
    chroma_collection_name: str = "translation_rag"
    retrieval_mode: str = "hybrid"
    hybrid_vector_weight: float = 0.6
    hybrid_bm25_weight: float = 0.4
    hybrid_rrf_k: int = 60
    hybrid_candidate_multiplier: int = 4
    function_call_max_rounds: int = 4
    function_call_max_tool_calls: int = 4
    function_call_allowed_tools: str = "get_rag_context"
    coding_workspace_root: str = "."
    coding_max_rounds: int = 6
    coding_max_tool_calls: int = 12
    coding_max_file_bytes: int = 128 * 1024
    coding_command_timeout_seconds: int = 30
    coding_sandbox_enabled: bool = True
    coding_sandbox_image: str = "agent-workbench-sandbox:py312"
    coding_sandbox_docker_binary: str = "docker"
    coding_sandbox_cpus: float = 1.0
    coding_sandbox_memory: str = "512m"
    coding_sandbox_pids_limit: int = 128
    coding_sandbox_output_bytes: int = 64 * 1024
    coding_sandbox_tmpfs_size: str = "64m"

    task_state_backend: str = "sqlite"
    task_state_db_path: str = "data/tasks.sqlite3"
    max_task_states: int = 10000

    max_upload_size_bytes: int = 2 * 1024 * 1024
    upload_allowed_extensions: str = ".txt,.md,.csv,.log"

    host: str = "127.0.0.1"
    port: int = 8000


settings = Settings()

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"
    chat_model: str = "qwen2.5-coder:7b"
    embedding_model: str = "nomic-embed-text"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rag_data_dir: Path = Path(".rag_data")
    max_repository_files: int = 5000
    github_clone_timeout_seconds: int = 180
    github_max_checkout_bytes: int = 250_000_000
    github_max_checkout_files: int = 10_000
    database_url: str = "sqlite:///.rag_data/application.db"
    redis_url: str = "redis://localhost:6379/0"
    indexing_worker_mode: str = "local"
    indexing_queue_name: str = "rag:index:queue"
    auth_enabled: bool = False
    oidc_issuer: str = "http://localhost:8080/realms/codebase-rag"
    oidc_audience: str = "codebase-rag"
    oidc_jwks_url: str | None = None
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"
    otel_enabled: bool = False
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_project: str = "engineering-codebase-rag"
    langsmith_workspace_id: str | None = None
    max_index_retries: int = 3
    max_citation_repair_attempts: int = 3
    max_retrieval_hops: int = 3
    max_retrieval_tool_calls: int = 6
    multi_query_count: int = 3
    parent_context_chunks: int = 1
    retrieval_candidate_k: int = 16
    rerank_candidate_k: int = 16
    retrieval_parallelism: int = 3
    index_batch_size: int = 64
    context_max_characters: int = 48_000
    context_max_chunks: int = 12
    context_max_chunks_per_file: int = 3
    query_plan_cache_ttl_seconds: int = 900
    embedding_cache_enabled: bool = True
    max_concurrent_model_requests: int = 2
    max_upload_bytes: int = 20_000_000
    temporary_upload_ttl_hours: int = 72
    malware_scan_enabled: bool = False
    clamav_host: str = "localhost"
    clamav_port: int = 3310
    clamav_timeout_seconds: float = 30
    graph_checkpoint_backend: str = "memory"
    graph_checkpoint_database_url: str | None = None
    rate_limit_per_minute: int = 120
    content_security_policy: str = ("default-src 'self'; img-src 'self' data:; "
                                    "style-src 'self' 'unsafe-inline'; script-src 'self' "
                                    "http://localhost:8080; connect-src 'self' "
                                    "http://127.0.0.1:* http://localhost:*")
    docs_content_security_policy: str = (
        "default-src 'self'; img-src 'self' data: https://fastapi.tiangolo.com; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; connect-src 'self'"
    )
    document_parser_provider: str = "local"
    vector_store_provider: str = "chroma"
    chroma_host: str | None = None
    chroma_port: int = 8000
    chroma_ssl: bool = False
    pinecone_api_key: str | None = None
    pinecone_control_host: str | None = None
    pinecone_index_host: str | None = None
    pinecone_index_name: str = "codebase-rag"
    pinecone_dimension: int = 768
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_auto_create: bool = False

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)

from backend.src.domain.enums import EmbedderType, SourceType


class ConnectorSettings(BaseSettings):
    """Configuration for a single connector (e.g., File, Tavily, Exa)."""

    id: int | None = 1
    name: str | None = "Local Files"
    origin_type: str | None = SourceType.FILESYSTEM

    # Optional fields (can be null/None in source data)
    api_key: SecretStr | None = None
    endpoint: str | None = None
    user_default_path: str | None = None
    last_indexed: str | None = None

    # Boolean fields
    is_indexable: bool = False
    periodic_indexing: bool = False

    model_config = SettingsConfigDict(env_prefix="CONNECTOR_")


class DatabaseSettings(BaseSettings):
    """Database configuration."""

    model_config = SettingsConfigDict(env_prefix="DB_")

    url: str = Field(
        default="postgresql+asyncpg://postgres:@localhost:5432/postgres",
        description="Database connection URL",
    )
    pool_size: int = Field(default=5, ge=1, le=50)
    pool_overflow: int = Field(default=10, ge=0, le=100)
    echo: bool = Field(default=False, description="Echo SQL statements")


class VectorSettings(BaseSettings):
    """Vector store configuration."""

    model_config = SettingsConfigDict(env_prefix="VECTOR_")
    collection_name: str = Field(default="test")
    vector_store_type: str = Field(default="QDRANT")


class QdrantSettings(BaseSettings):
    """Qdrant vector store configuration."""

    model_config = SettingsConfigDict(env_prefix="QDRANT_")

    host: str = Field(default="localhost")
    port: int = Field(default=6333, ge=1, le=65535)
    api_key: SecretStr | None = Field(default=None)
    url: str | None = Field(default=None, description="Full URL (overrides host/port)")
    collection_name: str = Field(default="default_collection")
    data_path: str | None = Field(
        default=None,
        description="Local path for embedded Qdrant persistence",
    )


class EmbeddingSettings(BaseSettings):
    """Embedding model configuration."""

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")

    model_type: EmbedderType = Field(default=EmbedderType.FASTEMBED)
    model_name: str = Field(default="BAAI/bge-m3")
    batch_size: int = Field(default=32, ge=1, le=256)
    normalize: bool = Field(default=True)


class RerankerSettings(BaseSettings):
    """Reranker configuration."""

    model_config = SettingsConfigDict(env_prefix="RERANKER_")

    enabled: bool = Field(default=False)
    model_name: str | None = Field(default=None)
    model_type: str | None = Field(default=None)
    top_k: int = Field(default=10, ge=1, le=100)


class FlashRankSettings(BaseSettings):
    """FlashRank reranker configuration."""

    model_config = SettingsConfigDict(env_prefix="FLASHRANK_")

    model_name: str = Field(default="ms-marco-MiniLM-L-12-v2")
    cache_dir: str = Field(default="./flashrank_cache")
    top_k: int = Field(default=10, ge=1, le=100)


class FalkorDBSettings(BaseSettings):
    """FalkorDB graph database configuration."""

    model_config = SettingsConfigDict(env_prefix="FALKORDB_")

    host: str = Field(default="localhost")
    port: int = Field(default=6379, ge=1, le=65535)
    graph_name: str = Field(default="fosra")


class ModelOpsSettings(BaseSettings):
    """Per-operation model selection: 'local' | 'api'."""

    model_config = SettingsConfigDict(env_prefix="MODELS_OPS_")

    query_expansion: str = Field(default="local")
    subagent: str = Field(default="local")
    generation: str = Field(default="api")
    classifier: str = Field(default="local")
    summarization: str = Field(default="local")
    code_embedding: str = Field(default="local")


class IngestionSettings(BaseSettings):
    """Ingestion pipeline configuration."""

    model_config = SettingsConfigDict(env_prefix="INGESTION_")

    chunk_size_parent: int = Field(default=768, ge=64, le=4096)
    chunk_size_child: int = Field(default=192, ge=32, le=1024)
    code_embedding_threshold: float = Field(default=0.4, ge=0.0, le=1.0)


class RetrievalSettings(BaseSettings):
    """Retrieval pipeline configuration."""

    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_")

    initial_summary_top_k: int = Field(default=20, ge=1, le=100)
    initial_direct_top_k: int = Field(default=10, ge=1, le=50)
    rerank_top_n: int = Field(default=15, ge=1, le=50)
    max_iterations: int = Field(default=5, ge=1, le=10)
    checklist_size: int = Field(default=5, ge=1, le=10)
    dense_weight: float = Field(default=3.0, ge=0.1, le=10.0)
    chunk_weight: float = Field(default=1.0, ge=0.1, le=10.0)
    rrf_k: int = Field(default=60, ge=1, le=200)
    feedback_a: float = Field(default=0.24, ge=0.0, le=1.0)
    feedback_b: float = Field(default=1.35, ge=0.0, le=5.0)
    feedback_c: float = Field(default=0.59, ge=0.0, le=1.0)


class AgentSettings(BaseSettings):
    """DeepAgent configuration."""

    model_config = SettingsConfigDict(env_prefix="AGENT_")

    max_retrieval_iterations: int = Field(default=3, ge=1, le=10)
    token_budget: int = Field(default=4096, ge=512, le=16384)
    fallback_model: str = Field(
        default="Qwen3.5-35B-A3B-Q4_K_M.gguf",
        description="Fallback model when no user preferences are configured",
    )
    fallback_api_base: str = Field(
        default="http://localhost:8045/v1",
        description="API base for fallback model",
    )


class CORSSettings(BaseSettings):
    """CORS configuration."""

    model_config = SettingsConfigDict(env_prefix="CORS_")

    allowed_origins: list[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:8000",
            "http://localhost:5173",
        ]
    )
    allowed_methods: list[str] = Field(default=["*"])
    allowed_headers: list[str] = Field(default=["*"])


class APIKeySettings(BaseSettings):
    """API keys and secrets."""

    model_config = SettingsConfigDict(env_prefix="")

    openrouter_api_key: SecretStr | None = Field(
        default=None, alias="OPENROUTER_API_KEY"
    )
    unstructured_api_key: SecretStr | None = Field(
        default=None, alias="UNSTRUCTURED_API_KEY"
    )
    secret_key: SecretStr | None = Field(default=None, alias="SECRET_KEY")


from backend.src.domain.enums import Environment

root_path = Path(__file__).resolve().parent.parent.parent.parent

config_path = root_path / "connector_settings.json"


# =============================================================================
# Main Settings Class
# =============================================================================
class Settings(BaseSettings):
    """Main application settings.

    All settings are loaded from environment variables and .env files.
    Nested settings use prefixes (e.g., DB_URL, QDRANT_HOST).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # Environment
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    debug: bool = Field(default=False)

    # Application
    app_name: str = Field(default="FOSRA")
    backend_url: str = Field(default="http://localhost:8000")
    global_vector_collection_name: str = "FOSRA"

    # Nested settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    falkordb: FalkorDBSettings = Field(default_factory=FalkorDBSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    flashrank: FlashRankSettings = Field(default_factory=FlashRankSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    api_keys: APIKeySettings = Field(default_factory=APIKeySettings)
    vectors: VectorSettings = Field(default_factory=VectorSettings)
    model_ops: ModelOpsSettings = Field(default_factory=ModelOpsSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    cors: CORSSettings = Field(default_factory=CORSSettings)

    @field_validator("environment", mode="before")
    @classmethod
    def validate_environment(cls, v: str | Environment) -> Environment:
        """Validate and normalize environment value."""
        if isinstance(v, Environment):
            return v
        return Environment(v.lower())

    @field_validator("agent", mode="before")
    @classmethod
    def validate_agent(
        cls, v: int | dict[str, Any] | AgentSettings | None
    ) -> AgentSettings | dict[str, Any]:
        """Handle AGENT env var conflict (e.g., AGENT=1 from system)."""
        if isinstance(v, AgentSettings):
            return v
        if isinstance(v, int):
            return {}
        return v if v else {}

    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == Environment.PRODUCTION

    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == Environment.DEVELOPMENT


# =============================================================================
# Helper Functions
# =============================================================================


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached settings instance.

    Use this function to get settings throughout the application.
    The settings are cached after first load.

    Returns:
        Settings instance
    """
    return Settings()


# Default settings instance for convenience
settings = get_settings()


# =============================================================================
# Re-export config classes from config.py
# =============================================================================
from backend.src.settings.config import (
    ChunkerConfig,
    CodeChunkerConfig,
    EmbedderConfig,
    LateChunkerConfig,
    LLMConfig,
    ModelPrefs,
    NeuralChunkerConfig,
    ParserConfig,
    PineconeConfig,
    QdrantConfig,
    RecursiveChunkerConfig,
    RerankerConfig,
    ScoredRetrieval,
    SemanticChunkerConfig,
    SentenceChunkerConfig,
    SlumberChunkerConfig,
    TokenChunkerConfig,
    UserPreferences,
    VectorStoreConfig,
)

__all__ = [
    "settings",
    "get_settings",
    "Settings",
    "APIKeySettings",
    "AgentSettings",
    "ConnectorSettings",
    "DatabaseSettings",
    "EmbeddingSettings",
    "FlashRankSettings",
    "FalkorDBSettings",
    "IngestionSettings",
    "ModelOpsSettings",
    "QdrantSettings",
    "RerankerSettings",
    "RetrievalSettings",
    "VectorSettings",
    "CORSSettings",
    "ChunkerConfig",
    "CodeChunkerConfig",
    "EmbedderConfig",
    "LateChunkerConfig",
    "LLMConfig",
    "ModelPrefs",
    "NeuralChunkerConfig",
    "ParserConfig",
    "PineconeConfig",
    "QdrantConfig",
    "RecursiveChunkerConfig",
    "RerankerConfig",
    "ScoredRetrieval",
    "SemanticChunkerConfig",
    "SentenceChunkerConfig",
    "SlumberChunkerConfig",
    "TokenChunkerConfig",
    "UserPreferences",
    "VectorStoreConfig",
]

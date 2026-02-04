from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


from FOSRABack.src.domain.enums import OriginType, EmbedderType


class ConnectorSettings(BaseSettings):
    """Configuration for a single connector (e.g., File, Tavily, Exa)."""

    id: int | None = 1
    name: str | None = "Local Files"
    origin_type: str | None = OriginType.FILESYSTEM

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


class EmbeddingSettings(BaseSettings):
    """Embedding model configuration."""

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")

    model_type: EmbedderType = Field(default=EmbedderType.FASTEMBED)
    model_name: str = Field(default="BAAI/bge-small-en-v1.5")
    batch_size: int = Field(default=32, ge=1, le=256)
    normalize: bool = Field(default=True)


class RerankerSettings(BaseSettings):
    """Reranker configuration."""

    model_config = SettingsConfigDict(env_prefix="RERANKER_")

    enabled: bool = Field(default=False)
    model_name: str | None = Field(default=None)
    model_type: str | None = Field(default=None)
    top_k: int = Field(default=10, ge=1, le=100)


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


from FOSRABack.src.domain.enums import Environment

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
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    api_keys: APIKeySettings = Field(default_factory=APIKeySettings)
    vectors: VectorSettings = Field(default_factory=VectorSettings)

    @field_validator("environment", mode="before")
    @classmethod
    def validate_environment(cls, v: str | Environment) -> Environment:
        """Validate and normalize environment value."""
        if isinstance(v, Environment):
            return v
        return Environment(v.lower())

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

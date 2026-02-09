from __future__ import annotations


from pydantic import Field, SecretStr
from pydantic.v1.utils import to_camel
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)

from backend.src.domain.enums import OriginType, EmbedderType

from pydantic import BaseModel, Field, SecretStr, ConfigDict


class _BaseModelFlex(BaseModel):
    """_BaseModelFlex with flexible config for attribute-based initialization."""

    _FLEXIBLE_CONFIG = ConfigDict(
        from_attributes=True,
        arbitrary_types_allowed=True,
        alias_generator=to_camel,
        populate_by_name=True,
    )

    model_config = _FLEXIBLE_CONFIG


class ConnectorSettingsRequest(_BaseModelFlex):
    """In / Out Settings for a single connector (e.g., File, Tavily, Exa)."""

    connector_id: int | None = 1
    connector_name: str | None = "Local Files"
    origin_type: str | None = OriginType.FILESYSTEM

    api_key: SecretStr | None = None
    endpoint: str | None = None
    user_default_path: str | None = None
    last_indexed: str | None = None

    is_indexable: bool = False
    periodic_indexing: bool = False

    model_config = SettingsConfigDict(
        populate_by_name=True,
    )


class DatabaseSettingsRequest(BaseSettings):
    """In / Out Database Settings."""

    model_config = SettingsConfigDict(env_prefix="DB_")

    url: str = Field(
        default="sqlite+aiosqlite:///:memory:",
        description="Database connection URL",
    )
    pool_size: int = Field(default=5, ge=1, le=50)
    pool_overflow: int = Field(default=10, ge=0, le=100)
    echo: bool = Field(default=False, description="Echo SQL statements")


class VectorSettingsRequest(BaseSettings):
    """In / Out Vector store Settings."""

    model_config = SettingsConfigDict(env_prefix="VECTOR_")
    collection_name: str = Field(default="test")
    vector_store_type: str = Field(default="QDRANT")


class QdrantSettingsRequest(BaseSettings):
    """In / Out Qdrant vector store settings."""

    model_config = SettingsConfigDict(env_prefix="QDRANT_")

    host: str = Field(default="localhost")
    port: int = Field(default=6333, ge=1, le=65535)
    api_key: SecretStr | None = Field(default=None)
    url: str | None = Field(default=None, description="Full URL (overrides host/port)")
    collection_name: str = Field(default="default_collection")


class EmbeddingSettingsRequest(BaseSettings):
    """In / Out Embedding model settings."""

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")

    model_type: EmbedderType = Field(default=EmbedderType.FASTEMBED)
    model_name: str = Field(default="BAAI/bge-small-en-v1.5")
    batch_size: int = Field(default=32, ge=1, le=256)
    normalize: bool = Field(default=True)


class RerankerSettingsRequest(BaseSettings):
    """In / Out Reranker settings."""

    model_config = SettingsConfigDict(env_prefix="RERANKER_")

    enabled: bool = Field(default=False)
    model_name: str | None = Field(default=None)
    model_type: str | None = Field(default=None)
    top_k: int = Field(default=10, ge=1, le=100)


class APIKeySettingsRequest(BaseSettings):
    """In / Out API keys and secrets."""

    model_config = SettingsConfigDict(env_prefix="")

    openrouter_api_key: SecretStr | None = Field(
        default=None, alias="OPENROUTER_API_KEY"
    )
    unstructured_api_key: SecretStr | None = Field(
        default=None, alias="UNSTRUCTURED_API_KEY"
    )
    secret_key: SecretStr | None = Field(default=None, alias="SECRET_KEY")

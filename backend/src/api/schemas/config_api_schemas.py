from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_litellm import ChatLiteLLM
from pydantic import BaseModel, Field, SecretStr, ConfigDict
from pydantic.v1.utils import to_camel

from backend.src.domain.enums import (
    EmbedderType,
    EmbeddingMode,
    ParserType,
    RerankerType,
    RetrievalMode,
    SearchStrategy,
    VectorStoreType,
    FileType,
)


if TYPE_CHECKING:
    pass


class _BaseModelFlex(BaseModel):
    """_BaseModelFlex with flexible config for attribute-based initialization."""

    _FLEXIBLE_CONFIG = ConfigDict(
        from_attributes=True,
        arbitrary_types_allowed=True,
        alias_generator=to_camel,
        populate_by_name=True,
        str_to_lower=True,
    )

    model_config = _FLEXIBLE_CONFIG


# =============================================================================
# Config Models
# =============================================================================
class BaseServiceConfig(_BaseModelFlex):
    """Common logic for all service configurations."""

    config_id: str | None = None
    config_name: str | None = None
    api_key: SecretStr | str | None = None
    api_base: str | None = None

    def get_key(self) -> str | None:
        """Centralized secret handling."""
        if isinstance(self.api_key, SecretStr):
            return self.api_key.get_secret_value()
        return self.api_key


class LLMConfigRequest(BaseServiceConfig):
    """User's LLM connection configuration."""

    provider: str = "openrouter"
    custom_provider: str | None = None
    model: str = "openai/gpt-3.5-turbo"
    language: str = "English"
    litellm_params: dict[str, Any] = {}

    def _llm_config_to_litellm(self) -> ChatLiteLLM:
        """Convert LLMConfigORM to typed config."""
        lite = ChatLiteLLM(
            model_name=self.model,
            api_key=str(self.api_key),
            api_base=self.api_base,
            **self.litellm_params,
        )
        return lite


class VectorStoreConfigRequest(BaseServiceConfig):
    """User's vector store connection configuration."""

    collection_name: str = "test"
    include_metadata: bool = True
    store_type: VectorStoreType | str = VectorStoreType.QDRANT
    host: str | None = None
    port: int | None = None
    location: str | None = ":memory:"
    url: str | None = None
    top_k: int = 10
    min_score: float = 0.0
    include_vectors: bool = False
    filter_conditions: dict[str, Any] = Field(default_factory=dict)

    def get_api_key_value(self) -> str | None:
        """Get the API key as a string."""
        if self.api_key is None:
            return None
        if isinstance(self.api_key, SecretStr):
            return self.api_key.get_secret_value()
        return self.api_key

    def to_init_config(self) -> dict[str, Any]:
        """Convert to initialization config dict for vector store."""
        config: dict[str, Any] = {
            "store_type": self.store_type,
            "host": self.host,
            "port": self.port,
            "collection_name": self.collection_name,
        }
        if self.url:
            config["url"] = self.url
        if self.api_key:
            config["api_key"] = self.get_api_key_value()
        return config


class EmbedderConfigRequest(BaseServiceConfig):
    """User's embedder connection configuration."""

    model: str = "BAAI/bge-small-en-v1.5"
    mode: EmbeddingMode = EmbeddingMode.DENSE_ONLY
    batch_size: int = 32
    max_concurrent: int = 3
    normalize: bool = True
    truncate: bool = True
    max_length: int = 512

    dense_model: str | None = None
    sparse_model: str | None = None
    late_model: str | None = None

    embedder_type: EmbedderType | str = EmbedderType.FASTEMBED

    def get_api_key_value(self) -> str | None:
        """Get the API key as a string."""
        if self.api_key is None:
            return None
        if isinstance(self.api_key, SecretStr):
            return self.api_key.get_secret_value()
        return self.api_key


class ParserConfigRequest(BaseServiceConfig):
    """User's parser configuration."""

    parser_type: ParserType | str = ParserType.DOCLING
    max_pages: int | None = None
    extract_tables: bool = True
    extract_images: bool = False
    ocr_enabled: bool = True
    language: str = "eng"
    timeout_seconds: int = 300
    fallback_parsers: list[ParserType] = Field(default_factory=list)
    generate_summary: bool = True

    model_config = {"from_attributes": True}

    def get_api_key_value(self) -> str | None:
        """Get the API key as a string."""
        if self.api_key is None:
            return None
        if isinstance(self.api_key, SecretStr):
            return self.api_key.get_secret_value()
        return self.api_key


class RerankerConfigRequest(BaseServiceConfig):
    """User's reranker configuration."""

    reranker_type: RerankerType | str = ""
    model: str | None = None
    enabled: bool = False
    params: dict[str, Any] | None = None
    top_k: int = 10
    score_threshold: float | None = None
    return_scores: bool = True
    batch_size: int = 32

    def get_api_key_value(self) -> str | None:
        """Get the API key as a string."""
        if self.api_key is None:
            return None
        if isinstance(self.api_key, SecretStr):
            return self.api_key.get_secret_value()
        return self.api_key


class RetrievalConfig(_BaseModelFlex):
    """Configuration for retrieval operations."""

    top_k: int = 10
    min_score: float = 0.0
    mode: RetrievalMode = RetrievalMode.CHUNKS
    strategy: SearchStrategy = SearchStrategy.VECTOR_ONLY

    file_types: list[FileType] | None = None
    source_ids: list[str] | None = None
    date_from: str | None = None
    date_to: str | None = None

    enable_rerank: bool = False
    rerank_top_k: int | None = None

    include_content: bool = True
    include_metadata: bool = True
    deduplicate: bool = True




from __future__ import annotations

from typing import TYPE_CHECKING, Any

from msgspec import field
from langchain_litellm import ChatLiteLLM
from loguru import logger
from pydantic import SecretStr

from backend.src.domain.schemas.file_schemas import StorageConfig
from backend.src.settings import settings
from backend.src.storage.utils.converters import DomainStruct
from ..enums import (
    ChunkerType,
    EmbedderType,
    EmbeddingMode,
    ParserType,
    RerankerType,
    VectorStoreType,
)


if TYPE_CHECKING:
    pass


class ChunkerConfig(DomainStruct):
    """Configuration for chunker behavior."""

    chunk_size: int = 512
    chunk_overlap: int = 128
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    # Semantic chunking specific
    similarity_threshold: float = 0.8
    embedding_model: str = "all-MiniLM-L6-v2"
    # Sentence chunking specific
    sentences_per_chunk: int = 5
    preferred_chunker_type: ChunkerType = ChunkerType.SEMANTIC


class LLMConfig(DomainStruct):
    """User's LLM connection configuration."""

    config_id: int = 0
    config_name: str = "default"
    provider: str = "openrouter"
    custom_provider: str | None = None
    model: str = "openai/gpt-3.5-turbo"
    api_key: str | SecretStr = SecretStr("secret")
    api_base: str = "https://openrouter.ai/api/v1"
    language: str = "English"
    litellm_params: dict[str, Any] = {}

    @property
    def model_name(self) -> str:
        """Alias for backward compatibility."""
        return self.model

    def get_api_key_value(self) -> str:
        """Get the API key as a string."""
        if isinstance(self.api_key, SecretStr):
            return self.api_key.get_secret_value()
        return self.api_key

    def _llm_config_to_litellm(self) -> ChatLiteLLM:
        """Convert LLMConfigORM to typed config."""
        lite = ChatLiteLLM(
            model_name=self.model,
            api_key=str(self.api_key),
            api_base=self.api_base,
            **self.litellm_params,
        )
        return lite


class VectorStoreConfig(DomainStruct):
    """User's vector store connection configuration."""

    config_id: int | None = None
    config_name: str | None = None
    api_key: SecretStr | None = None
    api_base: str | None = None
    collection_name: str = "test"
    include_metadata: bool = True
    distance_metric: str = ""
    store_type: VectorStoreType = VectorStoreType.QDRANT
    host: str | None = "localhost"
    port: int | None = None
    url: str | None = None
    vector_size: int = 384
    top_k: int = 10
    min_score: float = 0.0
    include_vectors: bool = False
    filter_conditions: dict[str, Any] = field(default_factory=dict)

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


class EmbedderConfig(DomainStruct):
    """User's embedder connection configuration."""

    config_id: int | None = None
    config_name: str | None = None
    model: str = "BAAI/bge-small-en-v1.5"
    api_key: SecretStr | None = None
    api_base: str | None = None
    mode: EmbeddingMode = EmbeddingMode.DENSE_ONLY
    batch_size: int = 32
    max_concurrent: int = 3
    normalize: bool = True
    truncate: bool = True
    max_length: int = 512

    dense_model: str | None = None
    sparse_model: str | None = None
    late_model: str | None = None

    embedder_type: EmbedderType = EmbedderType.FASTEMBED

    def get_api_key_value(self) -> str | None:
        """Get the API key as a string."""
        if self.api_key is None:
            return None
        if isinstance(self.api_key, SecretStr):
            return self.api_key.get_secret_value()
        return self.api_key


class ParserConfig(DomainStruct):
    """User's parser configuration."""

    config_id: int | None = None
    config_name: str | None = None
    preferrend_parser_type: ParserType = ParserType.UNSTRUCTURED
    api_key: SecretStr | None = None
    api_base: str | None = None
    max_pages: int | None = None
    extract_tables: bool = True
    extract_images: bool = False
    ocr_enabled: bool = True
    language: str = "eng"
    timeout_seconds: int = 300
    fallback_parsers: list[ParserType] = field(default_factory=list)
    generate_summary: bool = True

    def get_api_key_value(self) -> str | None:
        """Get the API key as a string."""
        if self.api_key is None:
            return None
        if isinstance(self.api_key, SecretStr):
            return self.api_key.get_secret_value()
        return self.api_key


class RerankerConfig(DomainStruct):
    """User's reranker configuration."""

    user_id: str = ""
    config_id: str = ""
    config_name: str = "New Reranker Config"
    reranker_type: RerankerType | None = None
    model: str | None = None
    api_key: SecretStr | None = None
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


# internal type checking -- serialize to and from dict on ingress and egress
# global settings -
class UserPreferences(DomainStruct, kw_only=True):
    """Domain Container for all typed user preferences."""

    # llm configs by role
    #
    llm_default: LLMConfig | None = None
    llm_fast: LLMConfig | None = None
    llm_logic: LLMConfig | None = None
    llm_heavy: LLMConfig | None = None

    # service configs
    parser: ParserConfig | None = None
    vector_store: VectorStoreConfig | None = None
    embedder: EmbedderConfig | None = None
    reranker: RerankerConfig | None = None
    chunker: ChunkerConfig | None = None

    def get_llm_config(self, role: str = "default") -> LLMConfig | None:
        """Get LLM config by role name."""
        role_map: dict[str, LLMConfig | None] = {
            "default": self.llm_default,
            "fast": self.llm_fast,
            "strategic": self.llm_logic,
            "heavy": self.llm_heavy,
        }
        config = role_map.get(role.lower())

        if not config:
            logger.warning(f"LLM config for role '{role}' not found. Using default.")
            return self.llm_default
        return config


class ModelPrefs(DomainStruct, kw_only=True):
    stream_chat_response: bool
    stream_delta_chunk_size: int
    seed: str
    stop_sequence: str
    temperature: int
    reasoning_effort: str
    logit_bias: str
    max_tokens: int
    top_k: int
    top_p: float
    min_p: float
    frequency_penalty: float
    presence_penalty: int
    mirostat: float
    mirostat_eta: float
    mirostat_tau: int
    repeat_last_n: int
    tfs_z: int
    repeat_penalty: float
    use_mmap: bool
    use_mlock: bool
    think_ollama: bool
    format_ollama: str
    num_keep_ollama: int
    num_ctx_ollama: int
    num_batch_ollama: int
    num_thread_ollama: int
    num_gpu_ollama: int
    keep_alive_ollama: bool


# workspace and convo settings - workspace -> convo precedence
class DynamicPrefs(DomainStruct, kw_only=True):
    llm_prefs: ModelPrefs | None = None  # remove none - set defaults
    search_enabled: bool = False
    rag_enabled: bool = True
    llm_override: LLMConfig | None = None

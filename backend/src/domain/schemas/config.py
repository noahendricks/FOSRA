from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from langchain_litellm import ChatLiteLLM
from langchain_qdrant import RetrievalMode
from loguru import logger
from msgspec import field
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic.v1.utils import to_camel

from ..enums import (
    ChunkerType,
    EmbedderType,
    EmbeddingMode,
    ParserType,
    RerankerType,
    VectorStoreType,
)


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


if TYPE_CHECKING:
    pass


class ChunkerConfig(BaseModel):
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


from chonkie import (
    BaseEmbeddings,
    CodeChunker,
    LateChunker,
    NeuralChunker,
    RecursiveChunker,
    RecursiveRules,
    SemanticChunker,
    SentenceTransformerEmbeddings,
    TokenizerProtocol,
)


class SemanticChunkerConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    embedding_model: Any = "minishlab/potion-base-32M"
    threshold: float = 0.8
    chunk_size: int = 2048
    similarity_window: int = 3
    min_sentences_per_chunk: int = 1
    min_characters_per_sentence: int = 24
    delim: str | list[str] = [". ", "! ", "? ", "\n"]
    include_delim: Literal["prev", "next"] | None = "prev"
    skip_window: int = 0
    filter_window: int = 5
    filter_polyorder: int = 3
    filter_tolerance: float = 0.2


class LateChunkerConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    embedding_model: Any = "nomic-ai/modernbert-embed-base"
    chunk_size: int = 2048
    rules: RecursiveRules = RecursiveRules()
    min_characters_per_chunk: int = 24


class RecursiveChunkerConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    tokenizer: Any = "character"
    chunk_size: int = 2048
    rules: RecursiveRules = RecursiveRules()
    min_characters_per_chunk: int = 24


class CodeChunkerConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    tokenizer: Any = "character"
    chunk_size: int = 2048
    language: Any = "auto"
    include_nodes: bool = False


class NeuralChunkerConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    model: Any = ""
    tokenizer: Any = None
    device_map: str = "auto"
    min_characters_per_chunk: int = 10
    stride: int | None = None


class LLMConfig(BaseModel):
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


class QdrantConfig(_BaseModelFlex):
    api_key: SecretStr | None = None
    api_base: str | None = None
    collection_name: str = "test"
    include_metadata: bool = True
    distance_metric: str = ""
    retrieval_mode: str = RetrievalMode.DENSE
    host: str = Field(default="localhost")
    port: int = Field(default=6333, ge=1, le=65535)
    url: str | None = None
    vector_size: int = 384
    top_k: int = 10
    min_score: float = 0.0
    include_vectors: bool = False
    filter_conditions: dict[str, Any] = Field(default_factory=dict)


from langchain_community.vectorstores import DistanceStrategy
from langchain_core.embeddings import Embeddings


class PineconeConfig(_BaseModelFlex):
    api_key: str | None = None
    index: Any | None = None
    embedding: Embeddings | None = None
    text_key: str | None = "text"
    namespace: str | None = None
    distance_strategy: DistanceStrategy | None = DistanceStrategy.COSINE
    index_name: str | None = None
    host: str | None = None
    dimension: int = 1536
    vector_size: int = 384
    top_k: int = 10
    min_score: float = 0.0
    include_vectors: bool = False
    include_values: bool = False
    include_meta: bool = False
    filter: dict[str, Any] = Field(default_factory=dict)


class VectorStoreConfig(BaseModel):
    config_id: int | None = None
    preferred_store: VectorStoreType = VectorStoreType.QDRANT
    qdrant_config: QdrantConfig = QdrantConfig()
    pinecone_config: PineconeConfig = PineconeConfig()
    # milvus_config: QdrantConfig = QdrantConfig
    # elasticsearch_config: QdrantConfig = QdrantConfig
    # opensearch_config: QdrantConfig = QdrantConfig


class EmbedderConfig(BaseModel):
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


class ParserConfig(BaseModel):
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


class RerankerConfig(BaseModel):
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
class UserPreferences(BaseModel):
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


class ModelPrefs(BaseModel):
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
class DynamicPrefs(BaseModel):
    llm_prefs: ModelPrefs | None = None  # remove none - set defaults
    search_enabled: bool = False
    rag_enabled: bool = True
    llm_override: LLMConfig | None = None

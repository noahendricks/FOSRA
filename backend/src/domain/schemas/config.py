from __future__ import annotations

import pathlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from chonkie.genie import BaseGenie, OpenAIGenie
from langchain_community.vectorstores import DistanceStrategy
from langchain_core.embeddings import Embeddings
from langchain_litellm import ChatLiteLLM
from langchain_qdrant import RetrievalMode
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


from chonkie import RecursiveRules, TokenizerProtocol


class SemanticChunkerConfig(_BaseModelFlex):
    model_config = {"arbitrary_types_allowed": True}

    embedding_model: Any = "nomic-ai/nomic-embed-text-v1.5"
    threshold: float = 0.4
    chunk_size: int = 512
    similarity_window: int = 2
    min_sentences_per_chunk: int = 2
    min_characters_per_sentence: int = 64
    delim: str | list[str] = [". ", "! ", "? ", "\n\n"]
    include_delim: Literal["prev", "next"] | None = "prev"
    skip_window: int = 1
    filter_window: int = 5
    filter_polyorder: int = 3
    filter_tolerance: float = 0.2
    trust_remote_code: bool = True


class LateChunkerConfig(_BaseModelFlex):
    model_config = {"arbitrary_types_allowed": True}

    embedding_model: Any = "nomic-ai/modernbert-embed-base"
    chunk_size: int = 2048
    min_characters_per_chunk: int = 24


class SlumberChunkerConfig(_BaseModelFlex):
    model_config = {"arbitrary_types_allowed": True}

    genie: BaseGenie | None = None  # lazy initialize
    tokenizer: str = "character"
    chunk_size: int = 1024
    candidate_size: int = 512
    min_characters_per_chunk: int = 24
    verbose: bool = True


class RecursiveChunkerConfig(_BaseModelFlex):
    model_config = {"arbitrary_types_allowed": True}

    tokenizer: Any = "character"
    chunk_size: int = 2048
    rules: RecursiveRules = RecursiveRules()
    min_characters_per_chunk: int = 24


class CodeChunkerConfig(_BaseModelFlex):
    model_config = {"arbitrary_types_allowed": True}

    tokenizer: Any = "character"
    chunk_size: int = 2048
    language: Any = "auto"
    include_nodes: bool = True


class NeuralChunkerConfig(_BaseModelFlex):
    model_config = {"arbitrary_types_allowed": True}

    model: Any = "mirth/chonky_modernbert_base_1"
    device_map: str = "auto"
    min_characters_per_chunk: int = 256
    stride: int | None = None


class TokenChunkerConfig(_BaseModelFlex):
    tokenizer: str = "character"
    chunk_size: int = 2048
    chunk_overlap: int | float = 0


class SentenceChunkerConfig(_BaseModelFlex):
    tokenizer: str = "character"
    chunk_size: int = 2048
    chunk_overlap: int = 0
    min_sentences_per_chunk: int = 1
    min_characters_per_sentence: int = 12
    approximate: bool = False
    delim: str | list[str] = [". ", "! ", "? ", "\n"]
    include_delim: Literal["prev", "next"] | None = "prev"


class ChunkerConfig(_BaseModelFlex):
    """Configuration for chunker behavior."""

    chunk_size: int = 2048
    max_inference_tokens: int = 8192
    max_levels: int = 3
    # HC200 fixed-size sub-chunking
    fixed_chunk_size: int = 250
    tokenizer: str = "auto"  # switch to rust based tokenizer model

    chunk_overlap: int = 256
    min_chunk_size: int = 100
    max_chunk_size: int = 2048

    slumber_config: SlumberChunkerConfig = SlumberChunkerConfig()
    neural_config: NeuralChunkerConfig = NeuralChunkerConfig()
    semantic_config: SemanticChunkerConfig = SemanticChunkerConfig()
    late_config: LateChunkerConfig = LateChunkerConfig()
    sentence_config: SentenceChunkerConfig = SentenceChunkerConfig()
    code_config: CodeChunkerConfig = CodeChunkerConfig()
    token_config: TokenChunkerConfig = TokenChunkerConfig()
    token_budget: int = 4096
    batch_size: int = 32

    embedding_model: str | Any = "nomic-ai/nomic-embed-text-v1.5"
    # Sentence chunking specific
    preferred_strategy: ChunkerType = ChunkerType.NEURAL


class LLMConfig(_BaseModelFlex):
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
    collection_name: str = "users_vectors"
    include_metadata: bool = True
    distance_metric: str = ""
    host: str = Field(default="localhost")
    port: int = Field(default=6333, ge=1, le=65535)
    url: str | None = None
    vector_size: int = 384
    top_k: int = 10
    min_score: float = 0.0
    include_vectors: bool = False
    filter_conditions: dict[str, Any] = Field(default_factory=dict)


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


class VectorStoreConfig(_BaseModelFlex):
    config_id: int | None = None
    preferred_store: VectorStoreType = VectorStoreType.QDRANT
    qdrant_config: QdrantConfig = QdrantConfig()
    pinecone_config: PineconeConfig = PineconeConfig()
    # milvus_config: QdrantConfig = QdrantConfig
    # elasticsearch_config: QdrantConfig = QdrantConfig
    # opensearch_config: QdrantConfig = QdrantConfig


def set_model_cache_dir():
    home = Path.home()
    new_path = home / "fosra_model_cache"
    new_path.mkdir(exist_ok=True, parents=True)
    return new_path


class ScoredRetrieval(_BaseModelFlex):
    rank: int | None = None
    score: float
    text: str
    doc_title: str
    chunk_id: str
    doc_id: str
    page_number: int
    start_index: int
    end_index: int


class EmbedderConfig(_BaseModelFlex):
    """User's embedder connection configuration."""

    config_id: int | None = None
    config_name: str | None = None
    api_key: SecretStr | None = None
    api_base: str | None = None
    mode: EmbeddingMode = EmbeddingMode.DENSE_ONLY
    batch_size: int = 32
    max_concurrent: int = 3
    normalize: bool = True
    truncate: bool = True
    max_length: int = 512
    cache_dir: Path = set_model_cache_dir()

    # default to fastembed
    embedder_type: EmbedderType = EmbedderType.FASTEMBED

    # default fastembed models
    dense_model: str = "nomic-ai/nomic-embed-text-v1.5"
    dense_dimensions: int = 768

    sparse_enabled: bool = True
    # !todo: add code models
    sparse_model: str | None = "prithivida/Splade_PP_en_v1"

    late_enabled: bool = True
    # !todo: add code models
    late_model: str | None = "colbert-ir/colbertv2.0"
    late_dimensions: int = 128

    cuda_enabled: bool = False

    def get_api_key_value(self) -> str | None:
        """Get the API key as a string."""
        if self.api_key is None:
            return None
        if isinstance(self.api_key, SecretStr):
            return self.api_key.get_secret_value()
        return self.api_key


class ParserConfig(_BaseModelFlex):
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


class RerankerConfig(_BaseModelFlex):
    """User's reranker configuration."""

    user_id: str = ""
    enabled: bool = True
    config_id: str = ""
    config_name: str = "New Reranker Config"
    api_key: SecretStr | None = None
    # default to FlashRank (CPU-only, no API key needed)
    RerankProvider: RerankerType = RerankerType.FLASHRANK
    model: str | None = "ms-marco-MiniLM-L-12-v2"
    top_k: int = 10
    score_threshold: float | None = None
    return_scores: bool = True
    batch_size: int = 32
    params: dict[str, Any] | None = None


# internal type checking -- serialize to and from dict on ingress and egress
# global settings -
class UserPreferences(_BaseModelFlex):
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


class ModelPrefs(_BaseModelFlex):
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
class DynamicPrefs(_BaseModelFlex):
    llm_prefs: ModelPrefs | None = None  # remove none - set defaults
    search_enabled: bool = False
    rag_enabled: bool = True
    llm_override: LLMConfig | None = None

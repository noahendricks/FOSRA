from enum import StrEnum


class Environment(StrEnum):
    """Application environment."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class VectorStoreType(StrEnum):
    """Supported vector store backends."""

    QDRANT = "QDRANT"
    PINECONE = "PINECONE"
    WEAVIATE = "WEAVIATE"
    MILVUS = "MILVUS"
    CHROMA = "CHROMA"
    PGVECTOR = "PGVECTOR"


class EmbedderType(StrEnum):
    """Types of embedders available."""

    FASTEMBED = "FASTEMBED"
    SENTENCE_TRANSFORMERS = "SENTENCE_TRANSFORMERS"
    OPENAI = "OPENAI"
    COHERE = "COHERE"
    VOYAGE = "VOYAGE"
    JINA = "JINA"
    MISTRAL = "MISTRAL"


class ParserType(StrEnum):
    """Types of parsers available."""

    DOCLING = "DOCLING"
    UNSTRUCTURED = "UNSTRUCTURED"
    MARKDOWN = "MARKDOWN"
    PYPDF = "PYPDF"
    LLAMA_PARSE = "LLAMA_PARSE"
    AZURE_DOCUMENT_INTELLIGENCE = "AZURE_DOCUMENT_INTELLIGENCE"
    GOOGLE_DOCUMENT_AI = "GOOGLE_DOCUMENT_AI"
    AWS_TEXTRACT = "AWS_TEXTRACT"


class RerankerType(StrEnum):
    """Types of rerankers available."""

    COHERE = "COHERE"
    CROSS_ENCODER = "CROSS_ENCODER"
    COLBERT = "COLBERT"
    BGE = "BGE"
    FLASHRANK = "FLASHRANK"

    # API rerankers
    JINA = "JINA"
    VOYAGE = "VOYAGE"


class EmbeddingMode(StrEnum):
    """Embedding modes - what types of vectors to generate."""

    DENSE_ONLY = "DENSE_ONLY"
    SPARSE_ONLY = "SPARSE_ONLY"
    HYBRID = "HYBRID"  # Dense + Sparse
    ADVANCED = "ADVANCED"  # Dense + Sparse + Late Interaction


class ChunkerType(StrEnum):
    """Types of chunkers available."""

    SEMANTIC = "SEMANTIC"
    TOKEN = "TOKEN"
    SENTENCE = "SENTENCE"
    FIXED = "FIXED"
    RECURSIVE = "RECURSIVE"


class MessageRole(StrEnum):
    """Message roles in chat conversations."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class OriginType(StrEnum):
    """Types of origins where information can be fetched from."""

    CRAWLED_URL = "CRAWLED_URL"
    FILESYSTEM = "FILESYSTEM"
    S3_DOCUMENT = "S3_DOCUMENT"
    YOUTUBE_VIDEO = "YOUTUBE_VIDEO"
    SLACK_MESSAGE = "SLACK_MESSAGE"
    NOTION_PAGE = "NOTION_PAGE"
    GITHUB_CONTENT = "GITHUB_CONTENT"
    LINEAR_ISSUE = "LINEAR_ISSUE"
    JIRA_ISSUE = "JIRA_ISSUE"
    DISCORD_MESSAGE = "DISCORD_MESSAGE"
    GMAIL_MESSAGE = "GMAIL_MESSAGE"
    LUMA_EVENT = "LUMA_EVENT"
    ELASTICSEARCH_DOCUMENT = "ELASTICSEARCH_DOCUMENT"
    EXTENSION_CONTENT = "EXTENSION_CONTENT"
    # Web search results from various connectors
    TAVILY_RESULT = "TAVILY_RESULT"
    SEARXNG_RESULT = "SEARXNG_RESULT"
    BAIDU_RESULT = "BAIDU_RESULT"
    LINKUP_RESULT = "LINKUP_RESULT"
    EXA_RESULT = "EXA_RESULT"
    SERPER_RESULT = "SERPER_RESULT"


class ConnectorType(StrEnum):
    """Types of connectors that retrieve information from external sources."""

    SERPER_API = "SERPER_API"
    EXA_API = "EXA_API"
    SEARXNG_API = "SEARXNG_API"
    TAVILY_API = "TAVILY_API"
    LINKUP_API = "LINKUP_API"
    BAIDU_API = "BAIDU_API"
    # Connector connectors
    JIRA_CONNECTOR = "JIRA_CONNECTOR"
    ELASTICSEARCH_CONNECTOR = "ELASTICSEARCH_CONNECTOR"
    SLACK_CONNECTOR = "SLACK_CONNECTOR"
    NOTION_CONNECTOR = "NOTION_CONNECTOR"
    GITHUB_CONNECTOR = "GITHUB_CONNECTOR"
    LINEAR_CONNECTOR = "LINEAR_CONNECTOR"
    DISCORD_CONNECTOR = "DISCORD_CONNECTOR"
    CONFLUENCE_CONNECTOR = "CONFLUENCE_CONNECTOR"
    GMAIL_CONNECTOR = "GMAIL_CONNECTOR"
    CALENDAR_CONNECTOR = "CALENDAR_CONNECTOR"
    AIRTABLE_CONNECTOR = "AIRTABLE_CONNECTOR"
    CLICKUP_CONNECTOR = "CLICKUP_CONNECTOR"
    LUMA_CONNECTOR = "LUMA_CONNECTOR"
    YOUTUBE_CONNECTOR = "YOUTUBE_CONNECTOR"
    FILE_UPLOAD = "FILE_UPLOAD"
    WEB_CRAWLER = "WEB_CRAWLER"
    EXTENSION = "EXTENSION"


class ChatType(StrEnum):
    QNA = "QNA"


class LiteLLMProvider(StrEnum):
    """Enum for LLM providers supported by LiteLLM."""

    OPENAI = "OPENAI"
    ANTHROPIC = "ANTHROPIC"
    GOOGLE = "GOOGLE"
    AZURE_OPENAI = "AZURE_OPENAI"
    BEDROCK = "BEDROCK"
    VERTEX_AI = "VERTEX_AI"
    GROQ = "GROQ"
    COHERE = "COHERE"
    MISTRAL = "MISTRAL"
    DEEPSEEK = "DEEPSEEK"
    XAI = "XAI"
    OPENROUTER = "OPENROUTER"
    TOGETHER_AI = "TOGETHER_AI"
    FIREWORKS_AI = "FIREWORKS_AI"
    REPLICATE = "REPLICATE"
    PERPLEXITY = "PERPLEXITY"
    OLLAMA = "OLLAMA"
    ALIBABA_QWEN = "ALIBABA_QWEN"
    MOONSHOT = "MOONSHOT"
    ZHIPU = "ZHIPU"
    ANYSCALE = "ANYSCALE"
    DEEPINFRA = "DEEPINFRA"
    CEREBRAS = "CEREBRAS"
    SAMBANOVA = "SAMBANOVA"
    AI21 = "AI21"
    CLOUDFLARE = "CLOUDFLARE"
    DATABRICKS = "DATABRICKS"
    COMETAPI = "COMETAPI"
    HUGGINGFACE = "HUGGINGFACE"
    CUSTOM = "CUSTOM"


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogStatus(StrEnum):
    IN_PROGRESS = "IN_PROGRESS"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class RAGStage(StrEnum):
    PROCESSING_QUERY = "PROCESSING_QUERY"
    RETRIEVING_QUERY = "RETRIEVING_QUERY"
    GENERATING_RESPONSE = "GENERATING_RESPONSE"
    STREAMING = "STREAMING"
    COMPLETE = "COMPLETE"


class LLMRole(StrEnum):
    HEAVY = "heavy"
    FAST = "fast"
    LOGICAL = "logical"


class StorageBackendType(StrEnum):
    FILESYSTEM = "FILESYSTEM"
    S3 = "S3"
    GCS = "GCS"
    AZURE_BLOB = "AZURE_BLOB"
    HTTP = "HTTP"
    FTP = "FTP"


class FileType(StrEnum):
    FILE = "FILE"
    CRAWLED_URL = "CRAWLED_URL"
    API_RESPONSE = "API_RESPONSE"
    UPLOADED = "UPLOADED"
    GENERATED = "GENERATED"
    IMAGE = "IMAGE"
    UNKNOWN = "UNKNOWN"


class ConversationStreamType(StrEnum):
    """Conversation streaming types."""

    CHAT = "chat"
    SEARCH = "search"
    HYBRID = "hybrid"
    ANALYTICAL = "analytical"


class RetrievalMode(StrEnum):
    """How to retrieve and return results."""

    CHUNKS = "CHUNKS"
    DOCUMENTS = "DOCUMENTS"
    HYBRID = "HYBRID"


class SearchStrategy(StrEnum):
    """Search strategy to use."""

    VECTOR_ONLY = "VECTOR_ONLY"
    KEYWORD_ONLY = "KEYWORD_ONLY"
    HYBRID = "HYBRID"
    MULTI_QUERY = "MULTI_QUERY"


from enum import StrEnum, auto


class DocumentType(StrEnum):
    """Supported document types based on official IANA Media Types."""

    # Applications
    PDF = "application/pdf"
    DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    DOC = "application/msword"
    PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    MANIFEST = "application/manifest+json"
    XML = "application/xml"

    # Text & Code
    HTML = "text/html"
    MARKDOWN = "text/markdown"
    TEXT = "text/plain"
    CSS = "text/css"
    CSV = "text/csv"
    JAVASCRIPT = "text/javascript"
    PYTHON = "text/x-python"
    RST = "text/x-rst"
    RTF = "application/rtf"
    RICHTEXT = "text/richtext"
    TSV = "text/tab-separated-values"
    VTT = "text/vtt"
    VCF = "text/x-vcard"

    UNKNOWN = "unknown"


class ConfigType(StrEnum):
    CHUNKER = "chunker"
    LLM = "llm"
    PARSER = "parser"
    VECTOR_STORE = "vector_store"
    EMBEDDER = "embedder"
    RERANKER = "reranker"


class ToolCategory(StrEnum):
    """Defines what 'kind' of tool this is."""

    LLM = "llm"
    VECTOR_STORE = "vector_store"
    EMBEDDER = "embedder"
    PARSER = "parser"
    STORAGE = "storage"
    RERANKER = "reranker"


class ConfigRole(StrEnum):
    """Defines what 'job' the tool performs in a context."""

    # LLM Roles
    PRIMARY_LLM = "primary_llm"
    FAST_LLM = "fast_llm"
    HEAVY_LLM = "heavy_llm"
    STRATEGIC_LLM = "strategic_llm"

    # Pipeline Roles
    DEFAULT_VECTOR_STORE = "default_vector_store"
    DEFAULT_EMBEDDER = "default_embedder"
    DEFAULT_PARSER = "default_parser"
    DEFAULT_RERANKER = "default_reranker"
    DEFAULT_STORAGE = "default_storage"

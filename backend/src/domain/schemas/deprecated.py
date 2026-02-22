

#WARN: DEPRECATED
# class ValidationResult(DomainStruct):
#     """Result of LLM configuration validation."""
#
#     is_valid: bool
#     error_message: str = ""
#     response_preview: str | None = None
#

#WARN: DEPRECATED
# class VectorPoint(DomainStruct):
#     id: str
#     payload: PayloadShape
#     dense_vector: list[float] | None = None
#     late_interaction_vectors: list[list[float]] | None = None
#
#WARN: DEPRECATED
# class EmbeddingResult(DomainStruct):
#     """Result of embedding operation."""
#     dense_vectors: list[list[float]]
#     late_interaction_vectors: list[list[list[float]]] | None = None
#     embedder_used: str = ""
#     embed_time_ms: float = 0.0
#     token_count: int = 0
#     errors: list[str] = field(default_factory=list)
#
#WARN: DEPRECATED
# class VectorSearchResult(DomainStruct):
#     """Standardized result from vector search."""
#     query_text: str
#     point_id: str
#     score: float
#     payload: dict[str, Any]
#     vector: list[float] | None = None
#
#WARN:DEPRECATED
# class ParsedDocument(DomainStruct):
#     """Result of document parsing."""
#     content: str
#     metadata: dict[str, Any] = field(default_factory=dict)
#     page_count: int = 1
#     parser_used: str = ""
#     parse_time_ms: float = 0.0
#     tables: list[dict[str, Any]] = field(default_factory=list)
#     images: list[dict[str, Any]] = field(default_factory=list)
#     errors: list[str] = field(default_factory=list



# WARN: DEPRECATED
# class StreamValidationResult(DomainStruct):
#     """Result of stream request validation."""
#     valid: bool
#     errors: list[str] = field(default_factory=list)
#     remediation: str | None = None
#     warnings: list[str] = field(default_factory=list)
#

#WARN: DEPRECATED
# class SourceGroup(DomainStruct):
#     """Grouped source with chunks for UI display."""
#     doc: Doc
#     chunks: list[Chunk]
#     top_score: float
#     chunk_count: int
#
#
# #WARN: DEPRECATED
# class PayloadShape(DomainStruct):
#     chunk_id: str
#     name: str
#     source_id: str
#     source_name: str
#     source_hash: str
#     origin_type: str
#     origin_path: str
#     file_type: str | None
#     chunk_text: str
#     start_index: int
#     end_index: int
#     token_count: int
#
# #WARN: DEPRECATED
# class RetrievedResult(DomainStruct):
#     """Schema for retrieved context from RAG."""
#     chunk_id: str
#     query_text: str
#     source_id: str
#     source_name: str | None = None
#     file_type: FileType | None = None
#     origin_type: str | None = None
#     similarity_score: float = 0.0
#     result_rank: int | None = None
#     reranker_score: float | None = None
#     retrieved_at: datetime = field(default_factory=utc_now)
#     model_used: str | None = None
#     source_snippet: str | None = field(
#         default=None,
#     )
#     contents: str = ""
#     metadata: dict[str, Any] = field(default_factory=dict)
#     def to_context_string(self) -> str:
#         """Format this retrieved context as a string for prompt construction."""
#         rank = self.result_rank if self.result_rank is not None else "N/A"
#         return f"[#{rank} RANKED CHUNK: FILE NAME: {self.source_name} CHUNK CONTENT: {self.contents}]"
#
#
# #WARN: DEPRECATED
# class RerankResult(DomainStruct):
#     """Result of reranking operation."""
#     documents: list[RetrievedResult]
#     reranker_used: str = ""
#     rerank_time_ms: float = 0.0
#     original_count: int = 0
#     filtered_count: int = 0
#     errors: list[str] = field(default_factory=list)
#
# #WARN: DEPRECATED
# class RetrievalConfig(DomainStruct):
#     """Configuration for retrieval operations."""
#     # Basic settings
#     top_k: int = 10
#     min_score: float = 0.0
#     mode: RetrievalMode = RetrievalMode.CHUNKS
#     strategy: SearchStrategy = SearchStrategy.VECTOR_ONLY
#     # Filtering
#     file_types: list[FileType] | None = None
#     source_ids: list[str] | None = None
#     date_from: str | None = None
#     date_to: str | None = None
#     # Reranking
#     enable_rerank: bool = False
#     rerank_top_k: int | None = None
#     # Advanced options
#     include_content: bool = True
#     include_metadata: bool = True
#     deduplicate: bool = True



# #WARN: DEPRECATED
# class StorageConfig(DomainStruct):
#     """Configuration for storage backend operations."""
#     timeout_seconds: int = 30
#     max_retries: int = 3
#     chunk_size: int = 8192
#     backend_options: dict[str, Any] = field(default_factory=dict)
#     preferred_backend_type: StorageBackendType = StorageBackendType.FILESYSTEM
#
# #WARN: DEPRECATED
# class FileMetadata(DomainStruct):
#     """Metadata about a file without content."""
#     metadata_hash: str | None = None
#     file_name: str = ""
#     document_type: DocType = DocType.UNKNOWN
#     origin_type: SourceType | None = None
#     size: int = 0
#     times_accessed: int | None = None
#     last_accessed: datetime = datetime.now()
#
# #WARN: DEPRECATED
# class FileContent(DomainStruct):
#     """Content retrieved from a file."""
#     file_path: str
#     file_hash: str
#     file_name: str
#     content: bytes | str
#     metadata: FileMetadata = field(default_factory=FileMetadata)
#
#
# #WARN: DEPRECATED
# class File(DomainStruct):
#     origin_path: str
#     uploaded_at: datetime = field(default_factory=utc_now)
#     name: str = ""
#     hash: str | None = None
#
#
# #WARN: DEPRECATED
# class FileProcessed(File):
#     source_content: str | None = None


from typing import Any

from pydantic.v1.utils import to_camel

from FOSRABack.src.domain.enums import ConversationStreamType

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


class ConversationStreamRequest(_BaseModelFlex):
    """Request for conversation streaming."""

    user_query: str
    langchain_chat_history: list[Any]
    search_mode: str = "chunks"
    document_ids_to_add_in_context: list[int] = Field(default_factory=list)
    language: str | None = None
    top_k: int = 10
    include_api_sources: bool = False
    origin_types_to_search: list[str] | None = None
    streaming_type: ConversationStreamType = ConversationStreamType.CHAT


class ConversationStreamResponse(_BaseModelFlex):
    """Response from conversation streaming."""

    stream_id: str
    user_query: str
    success: bool
    chunks_sent: int = 0
    stream_duration_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None


class ConversationStreamRequestModel(_BaseModelFlex):
    """Pydantic model for conversation streaming request."""

    user_query: str = Field(..., description="User query to process")
    langchain_chat_history: list[Any] = Field(
        default_factory=list, description="Chat history in LangChain format"
    )
    search_mode: str = Field(
        "chunks", description="Search mode (chunks, documents, hybrid)"
    )
    document_ids_to_add_in_context: list[int] = Field(
        default_factory=list, description="Document IDs to include in context"
    )
    language: str | None = Field(None, description="Language preference")
    top_k: int = Field(10, description="Number of results to retrieve")
    include_api_sources: bool = Field(
        False, description="Whether to include API sources"
    )
    origin_types_to_search: list[str] | None = Field(
        None, description="Origin types to search"
    )
    streaming_type: str = Field(
        "chat", description="Type of streaming (chat, search, hybrid, analytical)"
    )


class ConversationStreamResponseModel(_BaseModelFlex):
    """Pydantic model for conversation streaming response."""

    stream_id: str = Field(..., description="Unique stream identifier")
    user_query: str = Field(..., description="Original user query")
    success: bool = Field(..., description="Whether streaming succeeded")
    chunks_sent: int = Field(0, description="Number of chunks sent")
    stream_duration_ms: float | None = Field(
        None, description="Stream duration in milliseconds"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Streaming metadata"
    )
    error_message: str | None = Field(
        None, description="Error message if streaming failed"
    )

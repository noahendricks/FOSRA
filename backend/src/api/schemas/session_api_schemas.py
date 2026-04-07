from datetime import datetime
from typing import Any

from backend.src.api.schemas.message_schemas import AssistantMessage, UserMessage
from backend.src.api.schemas.base import BaseModelFlex
from backend.src.api.schemas.session_schemas import (
    SessionMetadataModel,
    SessionRevert,
    SessionTime,
)
from backend.src.api.schemas.source_api_schemas import (
    SourceResponseDeep,
    SourceResponseShallow,
)
from backend.src.domain.enums import ConversationStreamType, MessageRole
from backend.src.storage.utils.converters import utc_now
from pydantic import Field


class ConversationStreamRequest(BaseModelFlex):
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


class ConversationStreamResponse(BaseModelFlex):
    """Response from conversation streaming."""

    stream_id: str
    user_query: str
    success: bool
    chunks_sent: int = 0
    stream_duration_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None


class ConversationStreamRequestModel(BaseModelFlex):
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


class ConversationStreamResponseModel(BaseModelFlex):
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


# =============================================================================
# CONVERSATION CRUD SCHEMAS (moved from api_schemas.py)
# =============================================================================


class MessageAPI(BaseModelFlex):
    text: str
    session_id: str
    role: MessageRole
    user_id: str | None = None
    message_id: str | None = None
    message_metadata: dict[str, Any] | None = None


class SessionRequestBase(BaseModelFlex):
    user_id: str


class SessionRequest(SessionRequestBase):
    session_id: str


class NewSessionRequest(SessionRequestBase):
    title: str | None = "New Session"
    directory: str = ""
    version: str = "1"
    parent_id: str | None = None


class SessionDeleteRequest(SessionRequestBase):
    session_id: str
    session_list: list[str] | None = None


class SessionUpdateRequest(SessionRequestBase):
    title: str | None = "New Session"
    session_id: str
    session_metadata: dict[str, Any] | None = None
    messages: list[MessageAPI] | None = None
    data: dict[str, Any] | None = None
    directory: str | None = None
    version: str | None = None
    permission: dict[str, Any] | None = None
    revert: SessionRevert | None = None
    metadata: SessionMetadataModel | None = None


class SessionListItemResponse(SessionRequestBase):
    title: str
    session_id: str
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    message_count: int = Field(default=0)
    directory: str = ""
    version: str = "1"
    parent_id: str | None = None
    permission: dict[str, Any] | None = None
    revert: SessionRevert | None = None
    metadata: SessionMetadataModel | None = None


class NewSessionResponse(SessionRequestBase):
    session_id: str
    title: str | None = "New Session"
    session_metadata: dict[str, Any] | None = None
    directory: str = ""
    version: str = "1"
    parent_id: str | None = None
    time: SessionTime | None = None


class SessionFullResponse(SessionRequestBase):
    session_id: str
    title: str | None = "New Session"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    message_count: int = 0

    directory: str = ""
    version: str = "1"
    parent_id: str | None = None
    permission: dict[str, Any] | None = None
    revert: SessionRevert | None = None
    metadata: SessionMetadataModel | None = None
    time: SessionTime | None = None

    knowledge_sources: list[SourceResponseDeep | SourceResponseShallow] = Field(
        default_factory=list,
        description="All sources for this chat",
    )

    messages: list[UserMessage | AssistantMessage] = Field(default_factory=list)


class MessageUpdateRequest(BaseModelFlex):
    user_id: str
    session_id: str
    message_id: str
    role: str | None = None
    text: str | None = None
    message_metadata: dict[str, Any] | None = None

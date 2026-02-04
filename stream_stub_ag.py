"""Streaming Service

Provides business logic for conversation streaming operations.
Acts as a service layer between API routes and LangGraph orchestration.

Business logic includes:
- Conversation state initialization
- LangGraph streaming integration
- Stream chunk processing and formatting
- Error handling and completion
- Request validation
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

import logfire
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from ESRAmini.src.domain.exceptions import (
    ConvoError,
    ConvoStateError,
    StreamConnectionError,
    StreamProcessingError,
)
from ESRAmini.src.orchestration.main_flow.main_flow_config import (
    SearchMode as GraphSearchMode,
)
from ESRAmini.src.orchestration.main_flow.main_state import MainState
from ESRAmini.src.orchestration.main_flow.main_flow import app
from ESRAmini.src.services.streaming.formatter import StreamingFormatter

if TYPE_CHECKING:
    from ESRAmini.src.api.request_context import RequestContext
    from ESRAmini.src.domain.schemas import ConvoStreamRequest


class ConversationStreamingService:
    """Service layer for conversation streaming operations.

    All methods are static to follow the service pattern.
    Streaming operations are stateless per request.
    """

    @staticmethod
    @logfire.instrument(
        "Streaming conversation results",
        extract_args=True,
        span_name="Conversation Streaming",
    )
    async def stream_results(
        request: ConversationStreamRequest,
        session: AsyncSession,
    ) -> AsyncGenerator[str, None]:
        """Stream conversation results from LangGraph.

        Args:
            request: Conversation streaming request
            ctx: Request context with user/workspace info
            session: Database session

        Yields:
            Formatted stream chunks

        Raises:
            ConversationError: If conversation processing fails
            StreamConnectionError: If streaming connection fails
            StreamProcessingError: If stream chunk processing fails
        """
        streaming_formatter = StreamingFormatter()

        try:
            logger.info(
                f"Starting conversation stream for user {ctx.user_id}: "
                f"'{request.user_query[:50]}...'"
            )

            user_id_str = str(ctx.user_id)

            # Validate and convert search mode
            try:
                mode_enum = GraphSearchMode(request.search_mode)
            except ValueError:
                logger.warning(
                    f"Invalid search mode '{request.search_mode}', using default: CHUNKS"
                )
                mode_enum = GraphSearchMode.CHUNKS

            # Prepare configuration for LangGraph
            config = ConversationStreamingService._prepare_graph_config(
                request=request,
                user_id=user_id_str,
                workspace_id=ctx.workspace_id,
                search_mode=mode_enum,
            )

            # Initialize conversation state
            initial_state = ConversationStreamingService._initialize_conversation_state(
                request=request,
                ctx=ctx,
                session=session,
                streaming_formatter=streaming_formatter,
                mode_enum=mode_enum,
            )

            # Stream results from LangGraph
            async for chunk in ConversationStreamingService._stream_from_graph(
                initial_state=initial_state,
                config=config,
            ):
                yield chunk

            # Send completion signal
            yield streaming_formatter.format_completion()

            logger.success(
                f"Conversation stream completed for user {ctx.user_id}: "
                f"'{request.user_query[:50]}...'"
            )

        except asyncio.CancelledError:
            logger.warning(f"Conversation stream cancelled for user {ctx.user_id}")
            raise StreamConnectionError(
                operation="stream_results",
                connection_type="client",
                disconnect_reason="Client disconnected",
                remediation="Client may have closed the connection",
            ) from None
        except (
            ConversationError,
            StreamConnectionError,
            StreamProcessingError,
        ):
            raise
        except Exception as e:
            logger.error(f"Conversation streaming failed: {e}")
            raise ConversationError(
                operation="stream_results",
                user_query=request.user_query[:100] if request.user_query else "",
                reason=str(e),
                remediation="Check conversation configuration and LangGraph state",
            ) from e

    @staticmethod
    @logfire.instrument(
        "Preparing LangGraph configuration",
        extract_args=True,
        span_name="Graph Config Preparation",
    )
    def _prepare_graph_config(
        request: ConversationStreamRequest,
    ) -> dict[str, Any]:
        """Prepare configuration for LangGraph execution.

        Args:
            request: Conversation streaming request
            user_id: User ID string
            workspace_id: Workspace ID string
            search_mode: Search mode enum

        Returns:
            Configuration dictionary for LangGraph

        Raises:
            ConversationStateError: If configuration preparation fails
        """
        try:
            config = {
                "configurable": {
                    "user_query": request.user_query,
                    "user_id": user_id,
                    "workspace_id": workspace_id,
                    "search_mode": search_mode,
                    "document_ids_to_add_in_context": request.document_ids_to_add_in_context,
                    "language": request.language,
                    "top_k": request.top_k,
                    "include_api_sources": request.include_api_sources,
                    "origin_types_to_search": request.origin_types_to_search,
                    "enable_reranking": True,
                    "enable_citations": True,
                }
            }

            logger.debug(
                f"Prepared LangGraph config for search mode: {search_mode}, "
                f"top_k: {request.top_k}"
            )
            return config

        except Exception as e:
            logger.error(f"Failed to prepare graph configuration: {e}")
            raise ConversationStateError(
                operation="_prepare_graph_config",
                reason=str(e),
                remediation="Check conversation request parameters",
            ) from e

    @staticmethod
    @logfire.instrument(
        "Initializing conversation state",
        extract_args=True,
        span_name="Conversation State Initialization",
    )
    def _initialize_conversation_state(
        request: ConversationStreamRequest,
    ) -> MainState:
        """Initialize conversation state for LangGraph.

        Args:
            request: Conversation streaming request
            ctx: Request context
            session: Database session
            streaming_formatter: Streaming formatter instance
            mode_enum: Search mode enum

        Returns:
            Initialized MainState

        Raises:
            ConversationStateError: If state initialization fails
        """
        try:
            initial_state = MainState(
                ctx=ctx,
                db_session=session,
                streaming_service=streaming_formatter,
                chat_history=request.langchain_chat_history,
                vector_service=app_state.vector_service,
            )

            logger.debug(
                f"Initialized conversation state with "
                f"{len(request.langchain_chat_history)} history messages "
                f"and search mode: {mode_enum}"
            )
            return initial_state

        except Exception as e:
            logger.error(f"Failed to initialize conversation state: {e}")
            raise ConversationStateError(
                operation="_initialize_conversation_state",
                reason=str(e),
                remediation="Check conversation state parameters and dependencies",
            ) from e

    @staticmethod
    @logfire.instrument(
        "Streaming from LangGraph",
        extract_args=True,
        span_name="LangGraph Streaming",
    )
    async def _stream_from_graph(
        initial_state: MainState,
        config: dict[str, Any],
    ) -> AsyncGenerator[str, None]:
        """Stream results from LangGraph.

        Args:
            initial_state: Initialized conversation state
            config: LangGraph configuration

        Yields:
            Stream chunks

        Raises:
            StreamProcessingError: If streaming fails
        """
        chunks_yielded = 0

        try:
            logger.debug("Starting LangGraph streaming")

            async for chunk in app.astream(
                input=initial_state,
                config=config,
                stream_mode="custom",
            ):
                if isinstance(chunk, dict) and "yield_value" in chunk:
                    chunks_yielded += 1
                    yield chunk["yield_value"]

                    # Log progress every 10 chunks
                    if chunks_yielded % 10 == 0:
                        logger.debug(f"Yielded {chunks_yielded} stream chunks")

            logger.info(
                f"LangGraph streaming completed: {chunks_yielded} chunks yielded"
            )

        except Exception as e:
            logger.error(
                f"LangGraph streaming failed after {chunks_yielded} chunks: {e}"
            )
            raise StreamProcessingError(
                operation="_stream_from_graph",
                chunk_index=chunks_yielded,
                reason=str(e),
                remediation="Check LangGraph configuration and state transitions",
            ) from e

    @staticmethod
    @logfire.instrument(
        "Validating streaming request",
        extract_args=True,
        record_return=True,
        span_name="Stream Request Validation",
    )
    async def validate_stream_request(
        request: ConversationStreamRequest,
    ) -> dict[str, Any]:
        """Validate conversation streaming request.

        Args:
            request: Conversation streaming request
            ctx: Request context
            session: Database session

        Returns:
            Validation result with status and details
        """
        try:
            validation_errors = []

            # Validate user query
            if not request.user_query or not request.user_query.strip():
                validation_errors.append("User query is empty")

            # Validate search mode
            try:
                GraphSearchMode(request.search_mode)
            except ValueError:
                validation_errors.append(f"Invalid search mode: {request.search_mode}")

            # Validate top_k
            if request.top_k < 1 or request.top_k > 100:
                validation_errors.append(
                    f"top_k must be between 1 and 100, got {request.top_k}"
                )

            # Validate document IDs
            if request.document_ids_to_add_in_context:
                invalid_ids = [
                    id for id in request.document_ids_to_add_in_context if id < 0
                ]
                if invalid_ids:
                    validation_errors.append(f"Invalid document IDs: {invalid_ids}")

            if validation_errors:
                logger.warning(
                    f"Stream request validation failed for user {ctx.user_id}: "
                    f"{validation_errors}"
                )
                return {
                    "valid": False,
                    "errors": validation_errors,
                    "remediation": "Fix validation errors and retry",
                }

            logger.debug(f"Stream request validation passed for user {ctx.user_id}")
            return {
                "valid": True,
                "errors": [],
                "remediation": None,
            }

        except Exception as e:
            logger.error(f"Stream request validation failed: {e}")
            return {
                "valid": False,
                "errors": [str(e)],
                "remediation": "Check request parameters and try again",
            }

    @staticmethod
    @logfire.instrument("Getting streaming statistics")
    async def get_streaming_stats() -> dict[str, Any]:
        """Get streaming statistics and system status.

        Returns:
            Dictionary with streaming statistics
        """
        try:
            # TODO: Implement actual statistics collection
            # This would track active streams, throughput, etc.
            return {
                "active_streams": 0,
                "total_chunks_sent": 0,
                "average_stream_duration": 0.0,
                "system_status": "healthy",
            }

        except Exception as e:
            logger.error(f"Failed to get streaming stats: {e}")
            return {
                "active_streams": 0,
                "total_chunks_sent": 0,
                "average_stream_duration": 0.0,
                "system_status": "error",
                "error": str(e),
            }

    @staticmethod
    @logfire.instrument("Checking stream health", extract_args=True)
    async def check_stream_health(
        session: AsyncSession,
    ) -> dict[str, Any]:
        """Check streaming service health.

        Args:
            ctx: Request context
            session: Database session

        Returns:
            Health check results
        """
        try:
            # Basic health checks
            health_checks = {
                "database": False,
                "vector_service": False,
                "langgraph": False,
            }

            # Check database
            try:
                await session.execute("SELECT 1")
                health_checks["database"] = True
            except Exception as e:
                logger.error(f"Database health check failed: {e}")

            # Check vector service
            try:
                if app_state.vector_service:
                    health_checks["vector_service"] = True
            except Exception as e:
                logger.error(f"Vector service health check failed: {e}")

            # Check LangGraph app
            try:
                if app:
                    health_checks["langgraph"] = True
            except Exception as e:
                logger.error(f"LangGraph health check failed: {e}")

            all_healthy = all(health_checks.values())

            return {
                "healthy": all_healthy,
                "checks": health_checks,
                "status": "healthy" if all_healthy else "degraded",
            }

        except Exception as e:
            logger.error(f"Stream health check failed: {e}")
            return {
                "healthy": False,
                "checks": {},
                "status": "error",
                "error": str(e),
            }

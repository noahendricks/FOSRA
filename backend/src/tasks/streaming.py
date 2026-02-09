import asyncio
from typing import AsyncGenerator
from litellm import CustomStreamWrapper, completion, ModelResponse
import logfire
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from backend.src.api.request_context import RequestContext
from backend.src.api.schemas import LLMConfigRequest
from backend.src.api.schemas.convo_api_schemas import ConversationStreamRequest
from backend.src.domain.exceptions import (
    StreamConnectionError,
    StreamProcessingError,
    ConvoError,
)
from backend.src.domain.schemas import ConvoFull, LLMConfig, Message
from backend.src.services.conversation.formatter import StreamingFormatter

from .broker import broker


@logfire.instrument(
    "Streaming",
    extract_args=True,
    span_name="LangGraph Streaming",
)
@broker.task
async def get_stream(
    messages: list[Message],
    llm_config: LLMConfigRequest,
) -> AsyncGenerator[str, None] | ModelResponse:
    """Stream results from LangGraph."""
    chunks_yielded = 0

    try:
        logger.debug("Starting streaming")

        converted = [m.to_dict() for m in messages]

        response = completion(
            model=llm_config.model,
            stream=True,
            messages=converted,
            **llm_config.model_dump(),
        )

        if isinstance(response, CustomStreamWrapper):
            async for chunk in response:
                if isinstance(chunk, dict) and "choices" in chunk:
                    chunks_yielded += 1

                    choice = chunk["yield_value"]
                    content = choice["delta"]["content"]

                    yield content

                    # Log progress every 10 chunks
                    if chunks_yielded % 10 == 0:
                        logger.debug(f"Yielded {chunks_yielded} stream chunks")
        else:
            raise StreamProcessingError(reason="Not Stream")

        logger.info(f"Streaming completed: {chunks_yielded} chunks yielded")

    except Exception as e:
        logger.error(f"Streaming Task failed after {chunks_yielded} chunks: {e}")
        raise StreamProcessingError(
            chunk_index=chunks_yielded,
            reason=str(e),
            remediation="Check LangGraph configuration and state transitions",
        ) from e


@broker.task
async def stream_results(
    request: ConversationStreamRequest,
    session: AsyncSession,
    ctx: RequestContext,
    convo: ConvoFull,
) -> AsyncGenerator[str, None]:
    """Stream conversation results."""
    streaming_formatter = StreamingFormatter()

    try:
        logger.info(
            f"Starting conversation stream for user {ctx.user_id}: "
            f"'{request.user_query[:50]}...'"
        )

        user_id_str = str(ctx.user_id)

        stream = await get_stream(
            messages=convo.messages, llm_config=ctx.preferences.llm_default
        )
        # Send completion signal
        yield streaming_formatter.format_completion()

        logger.success(
            f"Conversation stream completed for user {ctx.user_id}: "
            f"'{request.user_query[:50]}...'"
        )

    except asyncio.CancelledError:
        logger.warning(f"Conversation stream cancelled for user {ctx.user_id}")
        raise StreamConnectionError(
            reason="Client disconnected",
            remediation="Client may have closed the connection",
        )
    except (
        ConvoError,
        StreamConnectionError,
        StreamProcessingError,
    ):
        raise
    except Exception as e:
        logger.error(f"Conversation streaming failed: {e}")
        raise ConvoError(
            reason=str(e),
            remediation="Check conversation configuration and LangGraph state",
        ) from e

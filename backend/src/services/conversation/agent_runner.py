"""
background agent execution with tui event emission.

runs the fosra agent as a background asyncio task and publishes
tui-shaped events (message.updated, message.part.delta, etc.)
through the global event bus.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage
from loguru import logger

from backend.src.api.events import event_bus
from backend.src.api.schemas.api_schemas import MessageResponse
from backend.src.api.schemas.tui_schemas import (
    AssistantMessage,
    AssistantMessagePath,
    AssistantMessageTime,
    AssistantMessageTokens,
    AssistantMessageTokensCache,
    DEFAULT_MODEL_ID,
    DEFAULT_PROVIDER_ID,
    PROJECT_DIR,
    PromptRequest,
    StepFinishPart,
    StepFinishPartTokens,
    StepFinishPartTokensCache,
    StepStartPart,
    TextPart,
    TextPartTime,
    ToolPart,
    ToolStateCompleted,
    ToolStateCompletedTime,
    ToolStateRunning,
    ToolStateRunningTime,
    UnknownError,
    UnknownErrorData,
    UserMessage,
    UserMessageModel,
    UserMessageTime,
)
from backend.src.domain.enums import MessageRole
from backend.src.services.conversation.agent_service import create_fosra_agent
from backend.src.services.conversation.conversation_service import ConversationService
from backend.src.storage.utils.converters import ulid_factory

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


def _unwrap_json_text(text: str) -> str:
    """strip json wrapping from llm output that should be plain text.

    some models wrap output in json like {"text": "..."} or {"code": "..."}.
    if the response is a json object with a single string value, return that value.
    """
    stripped = text.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return text
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            str_vals = [v for v in parsed.values() if isinstance(v, str)]
            if len(str_vals) == 1:
                return str_vals[0]
    except (json.JSONDecodeError, TypeError):
        pass
    return text


async def run_agent_with_events(
    session_id: str,
    user_id: str,
    prompt_request: PromptRequest,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """run the agent and emit tui events through the event bus.

    this function is meant to run as a background asyncio task,
    started by the /oc/session/{id}/prompt endpoint.
    """
    now = int(time.time())
    user_msg_id = prompt_request.messageID or ulid_factory()
    assistant_msg_id = ulid_factory()
    text_part_id = ulid_factory()
    step_start_id = ulid_factory()
    step_finish_id = ulid_factory()

    agent_name = prompt_request.agent or "fosra"

    user_text = ""
    for part in prompt_request.parts:
        if isinstance(part, dict):
            if part.get("type") == "text":
                text_val = part.get("text", "")
                if text_val:
                    user_text += text_val
        elif hasattr(part, "type") and part.type == "text" and hasattr(part, "text"):
            text_val = part.text
            if text_val:
                user_text += text_val

    if not user_text:
        logger.warning("Empty prompt received for session {}", session_id)
        return

    try:
        await event_bus.publish(
            {
                "type": "session.status",
                "properties": {
                    "sessionID": session_id,
                    "status": {"type": "busy"},
                },
            }
        )

        user_msg = UserMessage(
            id=user_msg_id,
            sessionID=session_id,
            role="user",
            time=UserMessageTime(created=now),
            agent=agent_name,
            model=UserMessageModel(
                providerID=DEFAULT_PROVIDER_ID,
                modelID=DEFAULT_MODEL_ID,
            ),
        ).model_dump()
        await event_bus.publish(
            {
                "type": "message.updated",
                "properties": {"info": user_msg},
            }
        )

        user_text_part = TextPart(
            id=ulid_factory(),
            sessionID=session_id,
            messageID=user_msg_id,
            type="text",
            text=user_text,
            time=TextPartTime(start=now, end=now),
        ).model_dump()
        await event_bus.publish(
            {
                "type": "message.part.updated",
                "properties": {"part": user_text_part},
            }
        )

        assistant_msg = AssistantMessage(
            id=assistant_msg_id,
            sessionID=session_id,
            role="assistant",
            time=AssistantMessageTime(created=now, completed=None),
            parentID=user_msg_id,
            modelID=DEFAULT_MODEL_ID,
            providerID=DEFAULT_PROVIDER_ID,
            mode="default",
            agent=agent_name,
            path=AssistantMessagePath(cwd=PROJECT_DIR, root=PROJECT_DIR),
            cost=1.0,
            tokens=AssistantMessageTokens(
                input=1,
                output=1,
                reasoning=1,
                cache=AssistantMessageTokensCache(read=1, write=0),
            ),
        ).model_dump()
        await event_bus.publish(
            {
                "type": "message.updated",
                "properties": {"info": assistant_msg},
            }
        )

        step_start = StepStartPart(
            id=step_start_id,
            sessionID=session_id,
            messageID=assistant_msg_id,
            type="step-start",
        ).model_dump()
        await event_bus.publish(
            {
                "type": "message.part.updated",
                "properties": {"part": step_start},
            }
        )

        text_part = TextPart(
            id=text_part_id,
            sessionID=session_id,
            messageID=assistant_msg_id,
            type="text",
            text="",
            time=TextPartTime(start=now, end=None),
        ).model_dump()
        await event_bus.publish(
            {
                "type": "message.part.updated",
                "properties": {"part": text_part},
            }
        )

        async with session_factory() as db_session:
            from backend.src.api.request_context import RequestContext

            ctx = await RequestContext.from_request(
                user_id=user_id,
                convo_id=session_id,
                session=db_session,
            )
            user_prefs = ctx.preferences

            await ConversationService.save_message(
                message=MessageResponse(
                    role=MessageRole.USER,
                    text=user_text,
                    user_id=user_id,
                    convo_id=session_id,
                    message_id=user_msg_id,
                ),
                convo_id=session_id,
                user_id=user_id,
                session=db_session,
            )

            backend = None
            if agent_name == "fosra-coding":
                from deepagents.backends import FilesystemBackend

                backend = FilesystemBackend(root_dir=PROJECT_DIR)

            agent, result_store = create_fosra_agent(user_prefs, backend=backend)

            lc_messages = [HumanMessage(content=user_text)]
            full_text = ""
            tool_call_parts: dict[str, dict[str, Any]] = {}

            logger.info("[STREAM] starting astream for session {}", session_id)
            chunk_count = 0
            async for msg, _metadata in agent.astream(
                {"messages": lc_messages},
                stream_mode="messages",
            ):
                chunk_count += 1
                logger.info(
                    "[STREAM] chunk #{} type={} content_type={} content_repr={}",
                    chunk_count,
                    type(msg).__name__,
                    type(getattr(msg, "content", None)).__name__,
                    repr(getattr(msg, "content", None))[:200],
                )
                if isinstance(msg, AIMessageChunk):
                    if msg.content:
                        text_chunk = (
                            msg.content
                            if isinstance(msg.content, str)
                            else str(msg.content)
                        )
                        full_text += text_chunk
                        await event_bus.publish(
                            {
                                "type": "message.part.delta",
                                "properties": {
                                    "sessionID": session_id,
                                    "messageID": assistant_msg_id,
                                    "partID": text_part_id,
                                    "field": "text",
                                    "delta": text_chunk,
                                },
                            }
                        )
                        logger.info(
                            "[STREAM] published delta len={} subscribers={}",
                            len(text_chunk),
                            event_bus.subscriber_count,
                        )

                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            call_id = str(tc.get("id", "")) or ulid_factory()
                            tool_name = tc.get("name", "unknown")
                            tool_args = tc.get("args", {})
                            part_id = ulid_factory()

                            tool_call_parts[call_id] = {
                                "part_id": part_id,
                                "tool": tool_name,
                                "start": time.time(),
                            }

                            tool_part = ToolPart(
                                id=part_id,
                                sessionID=session_id,
                                messageID=assistant_msg_id,
                                type="tool",
                                callID=call_id,
                                tool=tool_name,
                                state=ToolStateRunning(
                                    status="running",
                                    input=tool_args,
                                    title=tool_name,
                                    time=ToolStateRunningTime(start=int(time.time())),
                                ),
                            ).model_dump()
                            await event_bus.publish(
                                {
                                    "type": "message.part.updated",
                                    "properties": {"part": tool_part},
                                }
                            )

                elif isinstance(msg, ToolMessage):
                    call_id = getattr(msg, "tool_call_id", None)
                    if call_id and call_id in tool_call_parts:
                        tc_info = tool_call_parts[call_id]
                        end_time = int(time.time())

                        tool_part = ToolPart(
                            id=tc_info["part_id"],
                            sessionID=session_id,
                            messageID=assistant_msg_id,
                            type="tool",
                            callID=call_id,
                            tool=tc_info["tool"],
                            state=ToolStateCompleted(
                                status="completed",
                                input={},
                                output=str(msg.content)[:2000],
                                title=tc_info["tool"],
                                metadata={},
                                time=ToolStateCompletedTime(
                                    start=int(tc_info["start"]),
                                    end=end_time,
                                ),
                            ),
                        ).model_dump()
                        await event_bus.publish(
                            {
                                "type": "message.part.updated",
                                "properties": {"part": tool_part},
                            }
                        )

            logger.info(
                "[STREAM] loop finished. chunks={} full_text_len={}",
                chunk_count,
                len(full_text),
            )
            end_time = int(time.time())
            full_text = _unwrap_json_text(full_text)

            text_part_final = TextPart(
                id=text_part_id,
                sessionID=session_id,
                messageID=assistant_msg_id,
                type="text",
                text=full_text,
                time=TextPartTime(start=now, end=end_time),
            ).model_dump()
            await event_bus.publish(
                {
                    "type": "message.part.updated",
                    "properties": {"part": text_part_final},
                }
            )

            step_finish = StepFinishPart(
                id=step_finish_id,
                sessionID=session_id,
                messageID=assistant_msg_id,
                type="step-finish",
                reason="stop",
                cost=1.0,
                tokens=StepFinishPartTokens(
                    input=1,
                    output=1,
                    reasoning=1,
                    cache=StepFinishPartTokensCache(read=1, write=0),
                ),
            ).model_dump()
            await event_bus.publish(
                {
                    "type": "message.part.updated",
                    "properties": {"part": step_finish},
                }
            )

            assistant_msg_final = AssistantMessage(
                id=assistant_msg_id,
                sessionID=session_id,
                role="assistant",
                time=AssistantMessageTime(created=now, completed=end_time),
                parentID=user_msg_id,
                modelID=DEFAULT_MODEL_ID,
                providerID=DEFAULT_PROVIDER_ID,
                mode="default",
                agent=agent_name,
                path=AssistantMessagePath(cwd=PROJECT_DIR, root=PROJECT_DIR),
                cost=1.0,
                tokens=AssistantMessageTokens(
                    input=1,
                    output=1,
                    reasoning=1,
                    cache=AssistantMessageTokensCache(read=1, write=0),
                ),
                finish="stop",
            ).model_dump()
            await event_bus.publish(
                {
                    "type": "message.updated",
                    "properties": {"info": assistant_msg_final},
                }
            )

            sources_as_dicts = []
            if result_store.items:
                from backend.src.api.routes.workspace import _chunks_to_source_groups

                source_groups = _chunks_to_source_groups(result_store.items)  # type: ignore
                sources_as_dicts = [
                    group.model_dump(mode="json") for group in source_groups
                ]

            await ConversationService.save_message(
                message=MessageResponse(
                    role=MessageRole.ASSISTANT,
                    text=full_text,
                    user_id=user_id,
                    convo_id=session_id,
                    attached_sources=sources_as_dicts,
                ),
                convo_id=session_id,
                user_id=user_id,
                session=db_session,
            )

    except asyncio.CancelledError:
        logger.info("Agent task cancelled for session {}", session_id)
        await event_bus.publish(
            {
                "type": "session.status",
                "properties": {
                    "sessionID": session_id,
                    "status": {"type": "idle"},
                },
            }
        )
        return

    except Exception as e:
        logger.error("Agent error for session {}: {}", session_id, e)

        error_msg = AssistantMessage(
            id=assistant_msg_id,
            sessionID=session_id,
            role="assistant",
            time=AssistantMessageTime(created=now, completed=int(time.time())),
            parentID=user_msg_id,
            modelID=DEFAULT_MODEL_ID,
            providerID=DEFAULT_PROVIDER_ID,
            mode="default",
            agent=agent_name,
            path=AssistantMessagePath(cwd=PROJECT_DIR, root=PROJECT_DIR),
            cost=1.0,
            tokens=AssistantMessageTokens(
                input=1,
                output=1,
                reasoning=1,
                cache=AssistantMessageTokensCache(read=1, write=0),
            ),
            error=UnknownError(
                name="UnknownError",
                data=UnknownErrorData(message=str(e)),
            ),
        ).model_dump()
        await event_bus.publish(
            {
                "type": "message.updated",
                "properties": {"info": error_msg},
            }
        )

    finally:
        await event_bus.publish(
            {
                "type": "session.status",
                "properties": {
                    "sessionID": session_id,
                    "status": {"type": "idle"},
                },
            }
        )

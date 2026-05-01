"""
runner — composable agent execution with TUI event emission.

Exports the main run_agent_with_events function which orchestrates
event_formatter, permission_handler, stream_consumer, and utils.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage
from langgraph.graph.state import RunnableConfig
from loguru import logger as log

from backend.src.api.lifecycle import global_infra
from backend.src.api.schemas.tui_control_schemas import TextPart, UIMessage
from backend.src.services.session.agent_service import create_fosra_agent
from backend.src.services.session.conversation_service import SessionService
from backend.src.services.session.event_emitter import get_event_emitter
from backend.src.services.session.runner.event_formatter import EventFormatter
from backend.src.services.session.runner.prompt_extractor import (
    extract_provider_model,
    extract_user_text,
    resolve_llm_config,
)
from backend.src.services.session.runner.stream_consumer import (
    consume_stream,
)
from backend.src.services.session.runner.title_generator import (
    maybe_generate_title,
)
from backend.src.services.session.runner.utils import _unwrap_json_text
from backend.src.storage.utils.converters import ulid_factory

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


async def run_agent_with_events(
    session_id: str,
    user_id: str,
    prompt_request: Any,
    session_factory: async_sessionmaker[AsyncSession],
) -> Any:
    """Run the agent and emit TUI events through the event bus.

    This function is meant to run as a background asyncio task,
    started by the /oc/session/{id}/prompt endpoint.
    """
    now = int(time.time())
    user_msg_id = ulid_factory()
    assistant_msg_id = ulid_factory()
    step_finish_id = ulid_factory()

    agent_name = prompt_request.agent or "fosra"

    user_text = extract_user_text(prompt_request)

    slog = log.bind(session_id=session_id, user_id=user_id, agent=agent_name)

    if not user_text:
        slog.warning("empty_prompt")
        return

    # Extract provider/model info early for event emission
    provider_id, model_id = extract_provider_model(prompt_request)

    emitter: Any | None = None
    formatter: Any | None = None

    try:
        slog.info("session_start", user_text_len=len(user_text))
        emitter = get_event_emitter()
        await emitter.emit_busy(session_id)

        formatter = EventFormatter(emitter, session_id, now, provider_id, model_id)

        await formatter.emit_user_message(user_msg_id, agent_name, user_text)
        await formatter.emit_user_text_part(user_msg_id, user_text)

        text_part_id, step_start_id = await formatter.emit_assistant_message_start(
            assistant_msg_id, user_msg_id, agent_name
        )

        async with session_factory() as db_session:
            from backend.src.api.request_context import RequestContext

            ctx = await RequestContext.from_request(
                user_id=user_id,
                session_id=session_id,
                session=db_session,
            )
            user_prefs = ctx.preferences

            _ = await SessionService.save_message(
                message=UIMessage(
                    id=user_msg_id,
                    role="user",
                    parts=[TextPart(type="text", text=user_text)],
                    message_metadata={"parent_id": None, "root_id": None},
                ),
                session_id=session_id,
                user_id=user_id,
                session=db_session,
            )

            backend = None
            project_dir = (
                getattr(prompt_request, "project_dir", None) or "/home/roccoluxe/FOSRA"
            )
            if agent_name == "fosra-coding" or project_dir:
                from deepagents.backends import FilesystemBackend

                backend = FilesystemBackend(root_dir=project_dir)

            # build LLMConfig from TUI-selected provider/model
            llm_config = None
            if provider_id and model_id:
                llm_config = resolve_llm_config(provider_id, model_id, slog)

            agent, result_store = await create_fosra_agent(
                user_prefs,
                backend=backend,
                checkpointer=global_infra.checkpointer,
                llm_config=llm_config,
            )

            lc_messages = [HumanMessage(content=user_text)]
            cf = RunnableConfig(configurable={"thread_id": session_id})
            slog.info("astream_started")

            # -- BEGIN AGENT STREAM --
            astream_iterator = agent.astream(
                {"messages": lc_messages},
                config=cf,
                debug=True,
                stream_mode="messages",
                subgraphs=True,
            )

            result = await consume_stream(
                formatter,
                astream_iterator,
                assistant_msg_id,
                text_part_id,
            )

            full_text = result.full_text
            reasoning_part_id = result.reasoning_part_id
            reasoning_start_time = result.reasoning_start_time
            full_reasoning_text = result.full_reasoning_text

            slog.info(
                "stream_completed",
                chunk_count=result.chunk_count,
                full_text_len=len(full_text),
            )
            end_time = int(time.time())
            full_text = _unwrap_json_text(full_text)

            await formatter.emit_text_final(
                assistant_msg_id, text_part_id, full_text, end_time
            )

            if reasoning_part_id is not None and reasoning_start_time is not None:
                await formatter.emit_reasoning_final(
                    assistant_msg_id,
                    reasoning_part_id,
                    reasoning_start_time,
                    end_time,
                    full_reasoning_text,
                )

            await formatter.emit_step_finish(assistant_msg_id, step_finish_id)

            await formatter.emit_assistant_final(
                assistant_msg_id,
                user_msg_id,
                agent_name,
                end_time,
            )

            sources_as_dicts = []
            if result_store.items:
                from backend.src.api.routes._routes_utils import (
                    _chunks_to_source_groups,
                )
                from backend.src.services.retrieval.vector_service import RetrievedChunk

                chunks = [
                    RetrievedChunk(
                        text=item.content,
                        token_count=len(item.content.split()),
                        start_char=item.line_start,
                        score=item.score,
                        payload={
                            "source_id": item.file_id,
                            "end_char": item.line_end,
                        },
                    )
                    for item in result_store.items
                ]
                source_groups = _chunks_to_source_groups(chunks)
                sources_as_dicts = [
                    group.model_dump(mode="json") for group in source_groups
                ]

            _ = await SessionService.save_message(
                message=UIMessage(
                    id=assistant_msg_id,
                    role="assistant",
                    parts=[TextPart(type="text", text=full_text)],
                    message_metadata={"parent_id": user_msg_id, "root_id": user_msg_id},
                    sources=sources_as_dicts,
                ),
                session_id=session_id,
                user_id=user_id,
                session=db_session,
            )

            await maybe_generate_title(
                session_id=session_id,
                user_id=user_id,
                user_text=user_text,
                session_factory=session_factory,
                emitter=emitter,
                slog=slog,
            )

            return result

    except asyncio.CancelledError:
        slog.info("task_cancelled")
        if emitter is not None:
            await emitter.emit_session_status(session_id, {"type": "idle"})
        return None

    except Exception as e:
        slog.opt(exception=e).error("agent_error")
        from backend.src.api.schemas.tui_schemas import UnknownError, UnknownErrorData

        if formatter is not None:
            await formatter.emit_assistant_final(
                assistant_msg_id,
                user_msg_id,
                agent_name,
                int(time.time()),
                error=UnknownError(
                    name="UnknownError",
                    data=UnknownErrorData(message=str(e)),
                ),
            )
        if emitter is not None:
            await emitter.emit_session_error(session_id, str(e))

    finally:
        if emitter is not None:
            await emitter.emit_session_status(session_id, {"type": "idle"})

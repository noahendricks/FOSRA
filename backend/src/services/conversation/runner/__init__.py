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
from backend.src.api.schemas.api_schemas import MessageResponse
from backend.src.api.schemas.convo_api_schemas import ConvoUpdateRequest
from backend.src.domain.enums import MessageRole
from backend.src.services.conversation.agent_service import create_fosra_agent
from backend.src.services.conversation.conversation_service import ConversationService
from backend.src.services.conversation.runner.event_formatter import EventFormatter
from backend.src.services.conversation.runner.stream_consumer import consume_stream
from backend.src.services.conversation.runner.utils import _unwrap_json_text
from backend.src.services.session.event_emitter import get_event_emitter
from backend.src.settings import LLMConfig
from backend.src.storage.utils.converters import ulid_factory

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


_TOOL_NAME_MAP: dict[str, str] = {
    "BashTool": "bash",
    "Bash": "bash",
    "ReadTool": "read",
    "Read": "read",
    "WriteTool": "write",
    "Write": "write",
    "EditTool": "edit",
    "Edit": "edit",
    "GrepTool": "grep",
    "Grep": "grep",
    "GlobTool": "glob",
    "Glob": "glob",
    "ListTool": "list",
    "List": "list",
    "TaskTool": "task",
    "Task": "task",
    "WebFetchTool": "webfetch",
    "WebFetch": "webfetch",
    "WebSearchTool": "websearch",
    "WebSearch": "websearch",
    "CodeSearchTool": "codesearch",
    "CodeSearch": "codesearch",
    "TodoWrite": "todowrite",
    "TodoWriteTool": "todowrite",
    "QuestionTool": "question",
    "Question": "question",
    "ApplyPatch": "apply_patch",
    "ApplyPatchTool": "apply_patch",
    "SkillTool": "skill",
    "Skill": "skill",
}


def _normalize_tool_name(name: str) -> str:
    return _TOOL_NAME_MAP.get(name, name.lower())


def _resolve_llm_config(
    provider_id: str,
    model_id: str,
    slog: Any,
) -> LLMConfig | None:
    """Build LLMConfig from TUI-selected provider/model via provider registry."""
    import os

    from backend.src.api.schemas.provider_registry import _build_providers

    for provider in _build_providers():
        if provider.id != provider_id:
            continue
        model = provider.models.get(model_id)
        if not model:
            slog.warning("model_not_found", provider=provider_id, model=model_id)
            return None

        api_key = ""

        if provider.env:
            api_key = os.environ.get(provider.env[0], "")

        if not api_key and provider_id != "ollama":
            slog.warning("api_key_missing", provider=provider_id, env=provider.env)

        return LLMConfig(
            provider=provider_id,
            model=model.id,
            api_key=api_key or "not-set",
            api_base=model.api.url,
        )

    slog.warning("provider_not_found", provider=provider_id)
    return None


async def run_agent_with_events(
    session_id: str,
    user_id: str,
    prompt_request: Any,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Run the agent and emit TUI events through the event bus.

    This function is meant to run as a background asyncio task,
    started by the /oc/session/{id}/prompt endpoint.
    """
    now = int(time.time())
    user_msg_id = ulid_factory()
    assistant_msg_id = ulid_factory()
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

    slog = log.bind(session_id=session_id, user_id=user_id, agent=agent_name)

    if not user_text:
        slog.warning("empty_prompt")
        return

    # Extract provider/model info early for event emission
    provider_id = getattr(prompt_request, "providerID", None)
    model_id = getattr(prompt_request, "modelID", None)
    if not provider_id or not model_id:
        model_obj = getattr(prompt_request, "model", None)
        if model_obj:
            provider_id = getattr(model_obj, "providerID", None)
            model_id = getattr(model_obj, "modelID", None)

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

                backend = FilesystemBackend(
                    root_dir=prompt_request.project_dir or "/home/roccoluxe/FOSRA"
                )

            # build LLMConfig from TUI-selected provider/model
            llm_config = None
            if provider_id and model_id:
                llm_config = _resolve_llm_config(provider_id, model_id, slog)

            agent, result_store = create_fosra_agent(
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
                stream_mode="messages",
                subgraphs=True,
            )

            stats = await consume_stream(
                formatter,
                astream_iterator,
                assistant_msg_id,
                text_part_id,
            )

            full_text = stats["full_text"]
            reasoning_part_id = stats["reasoning_part_id"]
            reasoning_start_time = stats["reasoning_start_time"]

            slog.info(
                "stream_completed",
                chunk_count=stats["chunk_count"],
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
                from backend.src.api.routes.workspace import _chunks_to_source_groups

                source_groups = _chunks_to_source_groups(result_store.items)
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

            convo = await ConversationService.get_conversation_by_id(
                session=db_session,
                user_id=user_id,
                convo_id=session_id,
            )
            if convo.title == "New Convo" and user_text:
                words = user_text.split()
                title_words = words[:6]
                title = " ".join(title_words)
                filler = {
                    "hi",
                    "hello",
                    "hey",
                    "so",
                    "please",
                    "can",
                    "could",
                    "would",
                    "i",
                    "want",
                    "need",
                    "help",
                    "with",
                    "my",
                    "the",
                    "a",
                    "an",
                    "to",
                    "of",
                    "for",
                }
                first_word = title_words[0].lower().rstrip(".,!?;:")
                if first_word in filler and len(title_words) > 1:
                    for i, w in enumerate(title_words):
                        if w.lower().rstrip(".,!?;:") not in filler:
                            title = " ".join(title_words[i:])
                            break
                if len(title) > 50:
                    title = title[:47] + "..."
                title = title.rstrip(".,!?;:") or "New Convo"

                await ConversationService.update_conversation(
                    session=db_session,
                    convo_update=ConvoUpdateRequest(
                        user_id=user_id,
                        convo_id=session_id,
                        title=title,
                    ),
                )
                await emitter.emit_session_updated({"id": session_id, "title": title})
                slog.info("title_generated", new_title=title)

    except asyncio.CancelledError:
        slog.info("task_cancelled")
        await emitter.emit_session_status(session_id, {"type": "idle"})
        return

    except Exception as e:
        slog.error("agent_error", error=str(e), error_type=type(e).__name__)
        from backend.src.api.schemas.tui_schemas import UnknownError, UnknownErrorData

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
        await emitter.emit_session_error(session_id, str(e))

    finally:
        await emitter.emit_session_status(session_id, {"type": "idle"})

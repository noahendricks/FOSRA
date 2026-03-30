"""
background agent execution with tui event emission.

runs the fosra agent as a background asyncio task and publishes
tui-shaped events (message.updated, message.part.delta, etc.)
through the global event bus.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from typing import TYPE_CHECKING, Any

from loguru import logger as log
from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage
from langgraph.graph.state import RunnableConfig

from backend.src.api.lifecycle import global_infra
from backend.src.api.routes.oc.state import (
    ask_permission,
    ask_question,
    pending_permissions,
    pending_questions,
    session_diffs,
    session_todos,
)
from backend.src.api.schemas.convo_api_schemas import ConvoUpdateRequest
from backend.src.api.schemas.api_schemas import MessageResponse
from backend.src.api.schemas.tui_schemas import (
    DEFAULT_MODEL_ID,
    DEFAULT_PROVIDER_ID,
    PROJECT_DIR,
    AssistantMessage,
    AssistantMessagePath,
    AssistantMessageTime,
    AssistantMessageTokens,
    AssistantMessageTokensCache,
    PromptRequest,
    ReasoningPart,
    ReasoningPartTime,
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
from backend.src.services.session.event_emitter import get_event_emitter
from backend.src.storage.utils.converters import ulid_factory

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

event_emitter = get_event_emitter()


_TOOL_PERMISSIONS: dict[str, str] = {
    "read_file": "read",
    "write_file": "write",
    "edit_file": "edit",
    "execute": "bash",
    "BashTool": "bash",
    "bash": "bash",
    "TaskTool": "task",
    "task": "task",
    "WebFetchTool": "webfetch",
    "webfetch": "webfetch",
    "WebSearchTool": "websearch",
    "websearch": "websearch",
    "GrepTool": "grep",
    "grep": "grep",
    "GlobTool": "glob",
    "glob": "glob",
    "ListTool": "list",
    "ls": "list",
    "ReadTool": "read",
    "WriteTool": "write",
    "EditTool": "edit",
}

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
    """map llm tool names to tui-expected lowercase names."""
    return _TOOL_NAME_MAP.get(name, name.lower())


# TODO: VERY CRUDE -- should be llm generated
def _generate_title_from_text(text: str) -> str:
    """Generate a simple title from user message text.

    Takes the first few words of the message, cleans them up,
    and truncates to 50 characters.
    """
    if not text:
        return "New Convo"

    words = text.split()
    if not words:
        return "New Convo"

    # Take first 6 words
    title_words = words[:6]
    title = " ".join(title_words)

    # Clean up: remove common filler words at start
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
        # Try to find first non-filler word
        for i, w in enumerate(title_words):
            if w.lower().rstrip(".,!?;:") not in filler:
                title = " ".join(title_words[i:])
                break

    # Truncate at 50 chars
    if len(title) > 50:
        title = title[:47] + "..."

    # Clean up trailing punctuation
    title = title.rstrip(".,!?;:")

    return title if title else "New Convo"


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


def _parse_todo_output(content: str) -> list[dict[str, Any]]:
    """parse TodoWrite tool output into a list of Todo dicts."""
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and "todos" in parsed:
            return parsed["todos"]
    except (json.JSONDecodeError, TypeError):
        pass
    lines = content.strip().split("\n")
    todos = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("- [ ]"):
            todos.append(
                {"content": line[5:].strip(), "status": "pending", "priority": "medium"}
            )
        elif line.startswith("- [x]") or line.startswith("- [X]"):
            todos.append(
                {
                    "content": line[5:].strip(),
                    "status": "completed",
                    "priority": "medium",
                }
            )
        elif line.startswith("-"):
            todos.append(
                {"content": line[1:].strip(), "status": "pending", "priority": "medium"}
            )
    return todos


def _compute_git_diffs(project_dir: str) -> list[dict[str, Any]]:
    """compute git diffs for the project directory."""
    try:
        stat_out = subprocess.check_output(
            ["git", "diff", "--stat"],
            cwd=project_dir,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        full_out = subprocess.check_output(
            ["git", "diff"],
            cwd=project_dir,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        diffs = []
        for line in stat_out.strip().split("\n"):
            parts = line.split()
            if len(parts) >= 4 and parts[0] != "...":
                filename = parts[-4] if len(parts) >= 4 else parts[0]
                additions = 0
                deletions = 0
                for p in parts[-3:]:
                    if "+" in p:
                        additions += p.count("+")
                    if "-" in p:
                        deletions += p.count("-")
                diffs.append(
                    {
                        "file": filename,
                        "before": "",
                        "after": "",
                        "additions": additions,
                        "deletions": deletions,
                        "status": "modified",
                    }
                )
        return diffs
    except subprocess.CalledProcessError:
        return []


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
    user_msg_id = ulid_factory()
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

    slog = log.bind(session_id=session_id, user_id=user_id, agent=agent_name)

    if not user_text:
        slog.warning("empty_prompt")
        return

    try:
        slog.info("session_start", user_text_len=len(user_text))
        await event_emitter.emit_busy(session_id)

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
        await event_emitter.emit_message_updated(user_msg)
        slog.debug("user_message_emitted", message_id=user_msg_id)

        user_text_part = TextPart(
            id=ulid_factory(),
            sessionID=session_id,
            messageID=user_msg_id,
            type="text",
            text=user_text,
            time=TextPartTime(start=now, end=now),
        ).model_dump()
        await event_emitter.emit_message_part_updated(user_text_part)
        slog.debug("user_text_part_emitted")

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
        await event_emitter.emit_message_updated(assistant_msg)
        slog.debug("assistant_message_emitted", message_id=assistant_msg_id)

        step_start = StepStartPart(
            id=step_start_id,
            sessionID=session_id,
            messageID=assistant_msg_id,
            type="step-start",
        ).model_dump()
        await event_emitter.emit_message_part_updated(step_start)
        slog.debug("step_start_emitted")

        text_part = TextPart(
            id=text_part_id,
            sessionID=session_id,
            messageID=assistant_msg_id,
            type="text",
            text="",
            time=TextPartTime(start=now, end=None),
        ).model_dump()
        await event_emitter.emit_message_part_updated(text_part)
        slog.debug("streaming_text_part_created")

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
            slog.debug("user_message_saved_to_db", message_id=user_msg_id)

            backend = None
            if agent_name == "fosra-coding":
                from deepagents.backends import FilesystemBackend

                backend = FilesystemBackend(root_dir=PROJECT_DIR)

            agent, result_store = create_fosra_agent(
                user_prefs,
                backend=backend,
                checkpointer=global_infra.checkpointer,
            )

            lc_messages = [HumanMessage(content=user_text)]
            full_text = ""
            tool_call_parts: dict[str, dict[str, Any]] = {}
            handled_blocked_calls: set[str] = set()
            stream_abort = False
            reasoning_part_id: str | None = None
            reasoning_start_time: int | None = None

            slog.info("astream_started")
            chunk_count = 0
            cf = RunnableConfig(configurable={"thread_id": session_id})
            try:
                slog.debug(
                    "about_to_call_agent_astream", messages_count=len(lc_messages)
                )
                astream_iterator = agent.astream(
                    {"messages": lc_messages},
                    config=cf,
                    stream_mode="messages",
                    subgraphs=True,
                )
                slog.debug("astream_iterator_created")
                async for msg, _metadata in astream_iterator:
                    try:
                        if stream_abort:
                            break
                        chunk_count += 1
                        slog.debug(
                            "chunk_received",
                            chunk_count=chunk_count,
                            msg_type=type(msg).__name__,
                            content_type=type(getattr(msg, "content", None)).__name__,
                        )
                        if isinstance(msg, AIMessageChunk):
                            content_blocks = getattr(msg, "content_blocks", None)
                            if content_blocks:
                                for block in content_blocks:
                                    block_type = None
                                    try:
                                        block_type = (
                                            block.get("type")
                                            if isinstance(block, dict)
                                            else None
                                        )
                                        if block_type == "reasoning":
                                            reasoning_text = (
                                                block.get("reasoning", "") or ""
                                            )
                                            if reasoning_text:
                                                now_ts = int(time.time())
                                                if reasoning_part_id is None:
                                                    reasoning_part_id = ulid_factory()
                                                    reasoning_start_time = now_ts
                                                    slog.debug(
                                                        "reasoning_started",
                                                        reasoning_part_id=reasoning_part_id,
                                                    )
                                                    rp = ReasoningPart(
                                                        id=reasoning_part_id,
                                                        sessionID=session_id,
                                                        messageID=assistant_msg_id,
                                                        type="reasoning",
                                                        text="",
                                                        time=ReasoningPartTime(
                                                            start=now_ts, end=None
                                                        ),
                                                    ).model_dump()
                                                    await event_emitter.emit_message_part_updated(
                                                        rp
                                                    )
                                                slog.debug(
                                                    "about_to_emit_reasoning_delta"
                                                )
                                                await event_emitter.emit_message_part_delta(
                                                    session_id,
                                                    assistant_msg_id,
                                                    reasoning_part_id,
                                                    "text",
                                                    reasoning_text,
                                                )
                                        elif block_type == "text":
                                            text_val = block.get("text", "") or ""
                                            if text_val:
                                                log.info(
                                                    f"[AGENT RUNNER] text block delta: len={len(text_val)}, preview='{text_val[:50]}'"
                                                )
                                                full_text += text_val
                                                await event_emitter.emit_message_part_delta(
                                                    session_id,
                                                    assistant_msg_id,
                                                    text_part_id,
                                                    "text",
                                                    text_val,
                                                )
                                    except Exception as e:
                                        slog.error(
                                            "error_processing_content_block",
                                            block_type=block_type,
                                            error=str(e),
                                            error_type=type(e).__name__,
                                        )
                                        raise
                            elif msg.content:
                                text_chunk = (
                                    msg.content
                                    if isinstance(msg.content, str)
                                    else str(msg.content)
                                )
                                if text_chunk:
                                    log.info(
                                        f"[AGENT RUNNER] AIMessageChunk text delta: len={len(text_chunk)}, preview='{text_chunk[:50]}'"
                                    )
                                    full_text += text_chunk
                                    await event_emitter.emit_message_part_delta(
                                        session_id,
                                        assistant_msg_id,
                                        text_part_id,
                                        "text",
                                        text_chunk,
                                    )
                                    slog.debug(
                                        "text_delta_published",
                                        delta_len=len(text_chunk),
                                        subscriber_count=event_emitter._bus.subscriber_count,
                                    )

                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    call_id = None
                                    tool_name = None
                                    try:
                                        call_id = (
                                            str(tc.get("id", "")) or ulid_factory()
                                        )
                                        tool_name = tc.get("name", "unknown")
                                        tool_args = tc.get("args", {})
                                        part_id = ulid_factory()

                                        tool_call_parts[call_id] = {
                                            "part_id": part_id,
                                            "tool": _normalize_tool_name(tool_name),
                                            "tool_args": tool_args,
                                            "start": time.time(),
                                        }

                                        tool_part = ToolPart(
                                            id=part_id,
                                            sessionID=session_id,
                                            messageID=assistant_msg_id,
                                            type="tool",
                                            callID=call_id,
                                            tool=_normalize_tool_name(tool_name),
                                            state=ToolStateRunning(
                                                status="running",
                                                input=tool_args,
                                                title=_normalize_tool_name(tool_name),
                                                time=ToolStateRunningTime(
                                                    start=int(time.time())
                                                ),
                                            ),
                                        ).model_dump()
                                        await event_emitter.emit_message_part_updated(
                                            tool_part
                                        )
                                        slog.info(
                                            "tool_call_started",
                                            call_id=call_id,
                                            tool_name=_normalize_tool_name(tool_name),
                                        )

                                        tool_permission = _TOOL_PERMISSIONS.get(
                                            tool_name
                                        )
                                        if tool_permission:
                                            request_id, future, request = (
                                                ask_permission(
                                                    session_id=session_id,
                                                    permission=tool_permission,
                                                    patterns=[
                                                        str(tool_args.get("path", ""))
                                                    ]
                                                    if tool_args.get("path")
                                                    else [""],
                                                    metadata={
                                                        "tool": tool_name,
                                                        "args": tool_args,
                                                        "messageID": assistant_msg_id,
                                                    },
                                                    always=[],
                                                    tool={
                                                        "messageID": assistant_msg_id,
                                                        "callID": call_id,
                                                    },
                                                )
                                            )
                                            await event_emitter.emit_permission_asked(
                                                request
                                            )
                                            slog.info(
                                                "permission_requested",
                                                call_id=call_id,
                                                permission=tool_permission,
                                            )
                                            handled_blocked_calls.add(call_id)
                                            reply = await future
                                            slog.info(
                                                "permission_reply_received",
                                                call_id=call_id,
                                                reply=reply,
                                            )
                                            if reply == "reject":
                                                await event_emitter.emit_session_status(
                                                    session_id, {"type": "idle"}
                                                )
                                                stream_abort = True
                                                break
                                        elif tool_name == "Question":
                                            request_id, future, request = ask_question(
                                                session_id=session_id,
                                                questions=tool_args.get(
                                                    "questions", []
                                                ),
                                                tool={
                                                    "messageID": assistant_msg_id,
                                                    "callID": call_id,
                                                },
                                            )
                                            await event_emitter.emit_question_asked(
                                                request
                                            )
                                            slog.info(
                                                "question_asked",
                                                call_id=call_id,
                                                questions_count=len(
                                                    tool_args.get("questions", [])
                                                ),
                                            )
                                            handled_blocked_calls.add(call_id)
                                            reply = await future
                                            slog.info(
                                                "question_reply_received",
                                                call_id=call_id,
                                                reply=reply,
                                            )
                                            if reply == "reject":
                                                await event_emitter.emit_session_status(
                                                    session_id, {"type": "idle"}
                                                )
                                                stream_abort = True
                                                break
                                    except Exception as e:
                                        slog.error(
                                            "error_processing_tool_call",
                                            call_id=call_id
                                            if "call_id" in dir()
                                            else None,
                                            tool_name=tool_name
                                            if "tool_name" in dir()
                                            else None,
                                            error=str(e),
                                            error_type=type(e).__name__,
                                        )
                                        raise

                        elif isinstance(msg, ToolMessage):
                            call_id = None
                            try:
                                call_id = getattr(msg, "tool_call_id", None)
                                if (
                                    call_id
                                    and call_id in tool_call_parts
                                    and call_id not in handled_blocked_calls
                                ):
                                    tc_info = tool_call_parts[call_id]
                                    end_time = int(time.time())

                                    tool_name = tc_info["tool"]
                                    tool_args = tc_info.get("tool_args", {})
                                    metadata: dict[str, Any] = {}
                                    if tool_name == "todowrite":
                                        todos = _parse_todo_output(str(msg.content))
                                        metadata = {"todos": todos}
                                        session_todos[session_id] = todos
                                    tool_part = ToolPart(
                                        id=tc_info["part_id"],
                                        sessionID=session_id,
                                        messageID=assistant_msg_id,
                                        type="tool",
                                        callID=call_id,
                                        tool=tool_name,
                                        state=ToolStateCompleted(
                                            status="completed",
                                            input=tool_args,
                                            output=str(msg.content)[:2000],
                                            title=tool_name,
                                            metadata=metadata,
                                            time=ToolStateCompletedTime(
                                                start=int(tc_info["start"]),
                                                end=end_time,
                                            ),
                                        ),
                                    ).model_dump()
                                    await event_emitter.emit_message_part_updated(
                                        tool_part
                                    )
                                    slog.info(
                                        "tool_call_completed",
                                        call_id=call_id,
                                        tool_name=tool_name,
                                        duration_secs=round(
                                            end_time - int(tc_info["start"]), 2
                                        ),
                                    )

                                    new_text_part_id = ulid_factory()
                                    text_part = TextPart(
                                        id=new_text_part_id,
                                        sessionID=session_id,
                                        messageID=assistant_msg_id,
                                        type="text",
                                        text="",
                                        time=TextPartTime(
                                            start=int(time.time()), end=None
                                        ),
                                    ).model_dump()
                                    await event_emitter.emit_message_part_updated(
                                        text_part
                                    )
                                    text_part_id = new_text_part_id

                                    if tool_name == "todowrite":
                                        await event_emitter.emit_todo_updated(
                                            session_id, metadata.get("todos", [])
                                        )
                                        slog.debug(
                                            "todos_updated",
                                            session_id=session_id,
                                            todos_count=len(metadata.get("todos", [])),
                                        )
                            except Exception as e:
                                slog.error(
                                    "error_processing_tool_message",
                                    call_id=call_id if "call_id" in dir() else None,
                                    error=str(e),
                                    error_type=type(e).__name__,
                                )
                                raise
                    except Exception as e:
                        slog.error(
                            "error_in_stream_loop_chunk",
                            chunk_count=chunk_count,
                            msg_type=type(msg).__name__ if "msg" in dir() else None,
                            error=str(e),
                            error_type=type(e).__name__,
                        )
                        raise
            except Exception as e:
                slog.error(
                    "error_in_astream",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise

            slog.info(
                "stream_completed",
                chunk_count=chunk_count,
                full_text_len=len(full_text),
                tool_calls_count=len(tool_call_parts),
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
            await event_emitter.emit_message_part_updated(text_part_final)
            slog.debug(
                "text_part_final_emitted",
                part_id=text_part_id,
                text_len=len(full_text),
            )

            if reasoning_part_id is not None and reasoning_start_time is not None:
                reasoning_part_final = ReasoningPart(
                    id=reasoning_part_id,
                    sessionID=session_id,
                    messageID=assistant_msg_id,
                    type="reasoning",
                    text="",
                    time=ReasoningPartTime(start=reasoning_start_time, end=end_time),
                ).model_dump()
                await event_emitter.emit_message_part_updated(reasoning_part_final)
                slog.debug(
                    "reasoning_part_final_emitted",
                    reasoning_part_id=reasoning_part_id,
                    duration_secs=end_time - reasoning_start_time,
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
            await event_emitter.emit_message_part_updated(step_finish)
            slog.debug("step_finish_emitted", reason="stop")

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
            await event_emitter.emit_message_updated(assistant_msg_final)
            slog.debug(
                "assistant_message_final_emitted",
                message_id=assistant_msg_id,
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
            slog.debug(
                "assistant_message_saved_to_db",
                message_id=assistant_msg_id,
                sources_count=len(sources_as_dicts),
            )

            # Generate title from first user message if still default
            convo = await ConversationService.get_conversation_by_id(
                session=db_session,
                user_id=user_id,
                convo_id=session_id,
            )
            if convo.title == "New Convo" and user_text:
                new_title = _generate_title_from_text(user_text)
                await ConversationService.update_conversation(
                    session=db_session,
                    convo_update=ConvoUpdateRequest(
                        user_id=user_id,
                        convo_id=session_id,
                        title=new_title,
                    ),
                )
                await event_emitter.emit_session_updated(
                    {"id": session_id, "title": new_title}
                )
                slog.info("title_generated", old_title=convo.title, new_title=new_title)

            # NOTE: diffs not really a concern right now so bug can be deferred
            # diffs = _compute_git_diffs(PROJECT_DIR)
            # session_diffs[session_id] = diffs
            # await event_emitter.emit_session_diff(session_id, diffs)

    except asyncio.CancelledError:
        slog.info("task_cancelled")
        await event_emitter.emit_session_status(session_id, {"type": "idle"})
        return

    except Exception as e:
        slog.error("agent_error", error=str(e), error_type=type(e).__name__)

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
        await event_emitter.emit_message_updated(error_msg)
        await event_emitter.emit_session_error(session_id, error_msg["error"])
        slog.debug("error_message_emitted", error_name=error_msg["error"]["name"])

    finally:
        slog.debug("session_idle")
        await event_emitter.emit_session_status(session_id, {"type": "idle"})

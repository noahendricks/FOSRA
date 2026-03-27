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

from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage
from loguru import logger

from backend.src.api.events import event_bus
from backend.src.api.lifecycle import global_infra
from backend.src.api.routes.oc.state import (
    ask_permission,
    ask_question,
    pending_permissions,
    pending_questions,
    session_diffs,
    session_todos,
)
from backend.src.api.schemas.api_schemas import ConvoUpdateRequest, MessageResponse
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
from backend.src.storage.utils.converters import ulid_factory

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


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

            logger.info("[STREAM] starting astream for session {}", session_id)
            chunk_count = 0
            config = {"configurable": {"thread_id": session_id}}
            async for msg, _metadata in agent.astream(
                {"messages": lc_messages},
                config=config,
                stream_mode="messages",
            ):
                if stream_abort:
                    break
                chunk_count += 1
                logger.info(
                    "[STREAM] chunk #{} type={} content_type={} content_repr={}",
                    chunk_count,
                    type(msg).__name__,
                    type(getattr(msg, "content", None)).__name__,
                    repr(getattr(msg, "content", None))[:200],
                )
                if isinstance(msg, AIMessageChunk):
                    content_blocks = getattr(msg, "content_blocks", None)
                    if content_blocks:
                        for block in content_blocks:
                            block_type = (
                                block.get("type") if isinstance(block, dict) else None
                            )
                            if block_type == "reasoning":
                                reasoning_text = block.get("reasoning", "") or ""
                                if reasoning_text:
                                    now_ts = int(time.time())
                                    if reasoning_part_id is None:
                                        reasoning_part_id = ulid_factory()
                                        reasoning_start_time = now_ts
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
                                        await event_bus.publish(
                                            {
                                                "type": "message.part.updated",
                                                "properties": {"part": rp},
                                            }
                                        )
                                    await event_bus.publish(
                                        {
                                            "type": "message.part.delta",
                                            "properties": {
                                                "sessionID": session_id,
                                                "messageID": assistant_msg_id,
                                                "partID": reasoning_part_id,
                                                "field": "text",
                                                "delta": reasoning_text,
                                            },
                                        }
                                    )
                            elif block_type == "text":
                                text_val = block.get("text", "") or ""
                                if text_val:
                                    full_text += text_val
                                    await event_bus.publish(
                                        {
                                            "type": "message.part.delta",
                                            "properties": {
                                                "sessionID": session_id,
                                                "messageID": assistant_msg_id,
                                                "partID": text_part_id,
                                                "field": "text",
                                                "delta": text_val,
                                            },
                                        }
                                    )
                    elif msg.content:
                        text_chunk = (
                            msg.content
                            if isinstance(msg.content, str)
                            else str(msg.content)
                        )
                        if text_chunk:
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
                                    time=ToolStateRunningTime(start=int(time.time())),
                                ),
                            ).model_dump()
                            await event_bus.publish(
                                {
                                    "type": "message.part.updated",
                                    "properties": {"part": tool_part},
                                }
                            )

                            tool_permission = _TOOL_PERMISSIONS.get(tool_name)
                            if tool_permission:
                                request_id, future, request = ask_permission(
                                    session_id=session_id,
                                    permission=tool_permission,
                                    patterns=[
                                        str(tool_args.get("path", ""))
                                        if tool_args.get("path")
                                        else []
                                    ],
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
                                await event_bus.publish(
                                    {
                                        "type": "permission.asked",
                                        "properties": request,
                                    }
                                )
                                handled_blocked_calls.add(call_id)
                                reply = await future
                                if reply == "reject":
                                    await event_bus.publish(
                                        {
                                            "type": "session.status",
                                            "properties": {
                                                "sessionID": session_id,
                                                "status": {"type": "idle"},
                                            },
                                        }
                                    )
                                    stream_abort = True
                                    break
                            elif tool_name == "Question":
                                request_id, future, request = ask_question(
                                    session_id=session_id,
                                    questions=tool_args.get("questions", []),
                                    tool={
                                        "messageID": assistant_msg_id,
                                        "callID": call_id,
                                    },
                                )
                                await event_bus.publish(
                                    {
                                        "type": "question.asked",
                                        "properties": request,
                                    }
                                )
                                handled_blocked_calls.add(call_id)
                                reply = await future
                                if reply == "reject":
                                    await event_bus.publish(
                                        {
                                            "type": "session.status",
                                            "properties": {
                                                "sessionID": session_id,
                                                "status": {"type": "idle"},
                                            },
                                        }
                                    )
                                    stream_abort = True
                                    break

                elif isinstance(msg, ToolMessage):
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
                        await event_bus.publish(
                            {
                                "type": "message.part.updated",
                                "properties": {"part": tool_part},
                            }
                        )

                        new_text_part_id = ulid_factory()
                        text_part = TextPart(
                            id=new_text_part_id,
                            sessionID=session_id,
                            messageID=assistant_msg_id,
                            type="text",
                            text="",
                            time=TextPartTime(start=int(time.time()), end=None),
                        ).model_dump()
                        await event_bus.publish(
                            {
                                "type": "message.part.updated",
                                "properties": {"part": text_part},
                            }
                        )
                        text_part_id = new_text_part_id

                        if tool_name == "todowrite":
                            await event_bus.publish(
                                {
                                    "type": "todo.updated",
                                    "properties": {
                                        "sessionID": session_id,
                                        "todos": metadata.get("todos", []),
                                    },
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

            if reasoning_part_id is not None and reasoning_start_time is not None:
                reasoning_part_final = ReasoningPart(
                    id=reasoning_part_id,
                    sessionID=session_id,
                    messageID=assistant_msg_id,
                    type="reasoning",
                    text="",
                    time=ReasoningPartTime(start=reasoning_start_time, end=end_time),
                ).model_dump()
                await event_bus.publish(
                    {
                        "type": "message.part.updated",
                        "properties": {"part": reasoning_part_final},
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
                await event_bus.publish(
                    {
                        "type": "session.updated",
                        "properties": {"info": {"id": session_id, "title": new_title}},
                    }
                )

            diffs = _compute_git_diffs(PROJECT_DIR)
            session_diffs[session_id] = diffs
            await event_bus.publish(
                {
                    "type": "session.diff",
                    "properties": {"sessionID": session_id, "diff": diffs},
                }
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

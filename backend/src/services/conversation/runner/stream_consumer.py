"""
stream_consumer — consumes agent.astream() and emits events via EventFormatter.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessageChunk, ToolMessage
from loguru import logger as log

from backend.src.api.routes.oc.state import session_todos
from backend.src.services.conversation.runner.event_formatter import EventFormatter
from backend.src.services.conversation.runner.permission_handler import (
    handle_permission_request,
    handle_question_request,
)
from backend.src.services.conversation.runner.utils import _parse_todo_output
from backend.src.services.processing.loader_service import ulid_factory

if TYPE_CHECKING:
    pass


async def consume_stream(
    formatter: EventFormatter,
    astream_iterator: Any,
    assistant_msg_id: str,
    text_part_id: str,
) -> dict[str, Any]:
    """consume agent.astream() iterator, emit events, return stats."""
    full_text = ""
    tool_call_parts: dict[str, dict[str, Any]] = {}
    handled_blocked_calls: set[str] = set()
    reasoning_part_id: str | None = None
    reasoning_start_time: int | None = None
    full_reasoning_text = ""
    chunk_count = 0
    stream_abort = False
    # tracks whether we're inside a <think> block in raw content strings
    inside_think_block = False

    async for chunk_tuple in astream_iterator:
        if stream_abort:
            break
        chunk_count += 1

        log.debug(
            "stream_chunk_received raw",
            chunk_type=type(chunk_tuple).__name__,
            chunk_repr=repr(chunk_tuple)[:200],
        )

        # stream_mode="messages" yields tuples of (AIMessageChunk, metadata) or similar
        # actual structure: ((), (AIMessageChunk(...),)) - AIMessageChunk at chunk_tuple[1][0]
        if isinstance(chunk_tuple, (list, tuple)) and len(chunk_tuple) > 0:
            msg = None
            for item in chunk_tuple:
                if isinstance(item, AIMessageChunk):
                    msg = item
                    break
                if isinstance(item, (list, tuple)):
                    for sub_item in item:
                        if isinstance(sub_item, AIMessageChunk):
                            msg = sub_item
                            break
                    if msg:
                        break
            if (
                msg is None
                and len(chunk_tuple) >= 2
                and isinstance(chunk_tuple[1], (list, tuple))
                and len(chunk_tuple[1]) > 0
            ):
                if isinstance(chunk_tuple[1][0], AIMessageChunk):
                    msg = chunk_tuple[1][0]
        else:
            msg = chunk_tuple

        log.info(
            "stream_chunk_received parsed",
            msg_type=type(msg).__name__ if msg else "None",
            content_type=type(getattr(msg, "content", None)).__name__ if msg else "N/A",
            content=repr(getattr(msg, "content", None))[:100] if msg else "N/A",
            content_blocks=getattr(msg, "content_blocks", "N/A") if msg else "N/A",
        )

        if msg is None:
            log.warning(
                "stream_chunk: could not extract AIMessageChunk from",
                chunk_type=type(chunk_tuple).__name__,
            )
            continue

        if isinstance(msg, AIMessageChunk):
            # CONTENT PARSING
            # langchain_core's content_blocks property wraps unrecognized
            # content types (like Anthropic's "thinking") as "non_standard",
            # so we parse msg.content directly for reliable type detection.
            content = msg.content
            additional = msg.additional_kwargs or {}

            # reasoning via additional_kwargs (some providers use this)
            reasoning_from_kwargs = additional.get(
                "reasoning_content", ""
            ) or additional.get("reasoning_details", "")
            if reasoning_from_kwargs:
                now_ts = int(time.time())
                if reasoning_part_id is None:
                    reasoning_part_id = ulid_factory()
                    reasoning_start_time = now_ts
                    await formatter.emit_reasoning_start(
                        assistant_msg_id,
                        reasoning_part_id if reasoning_part_id else ulid_factory(),
                        now_ts,
                    )
                # model is actively reasoning — mark so string content
                # arriving without explicit <think> tags is also routed correctly
                inside_think_block = True
                full_reasoning_text += reasoning_from_kwargs
                await formatter.emit_reasoning_delta(
                    assistant_msg_id,
                    reasoning_part_id,
                    reasoning_from_kwargs,
                )

            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    block_type = block.get("type", "")
                    try:
                        if block_type in ("thinking", "reasoning"):
                            # "thinking" = Anthropic, "reasoning" = other providers
                            reasoning_text = (
                                block.get("thinking", "")
                                or block.get("reasoning", "")
                                or ""
                            )
                            if reasoning_text:
                                now_ts = int(time.time())
                                if reasoning_part_id is None:
                                    reasoning_part_id = ulid_factory()
                                    reasoning_start_time = now_ts
                                    await formatter.emit_reasoning_start(
                                        assistant_msg_id,
                                        reasoning_part_id,
                                        now_ts,
                                    )
                                full_reasoning_text += reasoning_text
                                await formatter.emit_reasoning_delta(
                                    assistant_msg_id,
                                    reasoning_part_id,
                                    reasoning_text,
                                )
                        elif block_type == "text":
                            text_val = block.get("text", "") or ""
                            if text_val:
                                full_text += text_val
                                await formatter.emit_text_delta(
                                    assistant_msg_id,
                                    text_part_id,
                                    text_val,
                                )
                    except Exception as e:
                        log.error(
                            "error_processing_content_block",
                            block_type=block_type,
                            error=str(e),
                        )
                        raise

            elif isinstance(content, str) and content:
                # PARSE <think>/</think> TAGS
                # ollama/qwen3 sends reasoning as raw <think>...</think>
                # in the content string. route these to reasoning part.
                remaining = content
                while remaining:
                    if inside_think_block:
                        close_idx = remaining.find("</think>")
                        if not reasoning_part_id:
                            reasoning_part_id = ulid_factory()
                        if close_idx != -1:
                            # everything before </think> is reasoning
                            reasoning_chunk = remaining[:close_idx]
                            remaining = remaining[close_idx + len("</think>") :]
                            inside_think_block = False
                            if reasoning_chunk:
                                full_reasoning_text += reasoning_chunk
                                await formatter.emit_reasoning_delta(
                                    assistant_msg_id,
                                    reasoning_part_id,
                                    reasoning_chunk,
                                )
                        else:
                            # entire remaining is reasoning
                            full_reasoning_text += remaining
                            await formatter.emit_reasoning_delta(
                                assistant_msg_id,
                                reasoning_part_id,
                                remaining,
                            )
                            remaining = ""
                    else:
                        open_idx = remaining.find("<think>")
                        if open_idx != -1:
                            # text before <think> is regular content
                            text_before = remaining[:open_idx]
                            remaining = remaining[open_idx + len("<think>") :]
                            inside_think_block = True
                            if text_before:
                                full_text += text_before
                                await formatter.emit_text_delta(
                                    assistant_msg_id,
                                    text_part_id,
                                    text_before,
                                )
                            # ensure reasoning part exists
                            if reasoning_part_id is None:
                                now_ts = int(time.time())
                                reasoning_part_id = ulid_factory()
                                reasoning_start_time = now_ts
                                await formatter.emit_reasoning_start(
                                    assistant_msg_id,
                                    reasoning_part_id,
                                    now_ts,
                                )
                        else:
                            # no tags — regular text
                            full_text += remaining
                            await formatter.emit_text_delta(
                                assistant_msg_id,
                                text_part_id,
                                remaining,
                            )
                            remaining = ""

            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    call_id = None
                    tool_name = None
                    try:
                        call_id = (
                            str(tc.get("id", ""))
                            or __import__(
                                "backend.src.storage.utils.converters",
                                fromlist=["ulid_factory"],
                            ).ulid_factory()
                        )
                        tool_name = tc.get("name", "unknown")
                        tool_args = tc.get("args", {})
                        part_id = __import__(
                            "backend.src.storage.utils.converters",
                            fromlist=["ulid_factory"],
                        ).ulid_factory()

                        tool_call_parts[call_id] = {
                            "part_id": part_id,
                            "tool": tool_name,
                            "tool_args": tool_args,
                            "start": time.time(),
                        }

                        await formatter.emit_tool_start(
                            assistant_msg_id,
                            call_id,
                            tool_name,
                            tool_args,
                        )

                        if tool_name == "Question":
                            handled, abort = await handle_question_request(
                                formatter._emitter,
                                formatter._session_id,
                                assistant_msg_id,
                                call_id,
                                tool_args,
                            )
                            if handled:
                                handled_blocked_calls.add(call_id)
                                if abort:
                                    stream_abort = True
                                    break
                        else:
                            handled, abort = await handle_permission_request(
                                formatter._emitter,
                                formatter._session_id,
                                assistant_msg_id,
                                call_id,
                                tool_name,
                                tool_args,
                            )
                            if handled:
                                handled_blocked_calls.add(call_id)
                                if abort:
                                    stream_abort = True
                                    break
                    except Exception as e:
                        log.error("error_processing_tool_call", error=str(e))
                        raise

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
                    session_todos[formatter._session_id] = todos

                new_text_part_id = await formatter.emit_tool_end(
                    assistant_msg_id,
                    call_id,
                    tool_name,
                    tool_args,
                    str(msg.content),
                    tc_info["start"],
                    metadata,
                )

                if tool_name == "todowrite":
                    await formatter._emitter.emit_todo_updated(
                        formatter._session_id, metadata.get("todos", [])
                    )

    # safety net: strip any stray <think>/</think> tags from saved text
    full_text = re.sub(r"</?think>", "", full_text)

    return {
        "full_text": full_text,
        "tool_call_parts": tool_call_parts,
        "reasoning_part_id": reasoning_part_id,
        "reasoning_start_time": reasoning_start_time,
        "full_reasoning_text": full_reasoning_text,
        "chunk_count": chunk_count,
    }

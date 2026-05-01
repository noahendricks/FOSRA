"""
stream_consumer — consumes agent.astream() and emits events via EventFormatter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessageChunk, ToolMessage
from loguru import logger as log

from backend.src.services.session.runner.content_parser import (
    ParseState,
    parse_chunk_content,
)
from backend.src.services.session.runner.event_formatter import EventFormatter
from backend.src.services.session.runner.tool_handler import (
    ToolCallRecord,
    handle_tool_calls,
    handle_tool_result,
)

if TYPE_CHECKING:
    pass


@dataclass
class StreamResult:
    full_text: str
    tool_call_parts: dict[str, ToolCallRecord]
    reasoning_part_id: str | None
    reasoning_start_time: int | None
    full_reasoning_text: str
    chunk_count: int


def extract_ai_message(chunk_tuple: Any) -> Any:
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
    return msg


async def consume_stream(
    formatter: EventFormatter,
    astream_iterator: Any,
    assistant_msg_id: str,
    text_part_id: str,
) -> StreamResult:
    """consume agent.astream() iterator, emit events, return StreamResult."""

    state = ParseState()
    tool_call_parts: dict[str, ToolCallRecord] = {}
    handled_blocked_calls: set[str] = set()
    chunk_count = 0
    stream_abort = False

    async for chunk_tuple in astream_iterator:
        if stream_abort:
            break
        chunk_count += 1

        log.debug(
            "stream_chunk_received raw",
            chunk_type=type(chunk_tuple).__name__,
            chunk_repr=repr(chunk_tuple)[:200],
        )

        msg = extract_ai_message(chunk_tuple)

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
            deltas = parse_chunk_content(msg, state)

            for delta in deltas:
                log.info(
                    f"[stream-consumer] delta kind {delta.kind}, content {delta.content}"
                )
                if delta.kind == "text":
                    await formatter.emit_text_delta(
                        assistant_msg_id,
                        text_part_id,
                        delta.content,
                    )
                elif delta.kind == "reasoning_start":
                    assert delta.reasoning_part_id is not None
                    assert delta.reasoning_start_time is not None
                    await formatter.emit_reasoning_start(
                        assistant_msg_id,
                        delta.reasoning_part_id,
                        delta.reasoning_start_time,
                    )
                elif delta.kind == "reasoning_delta":
                    assert delta.reasoning_part_id is not None
                    await formatter.emit_reasoning_delta(
                        assistant_msg_id,
                        delta.reasoning_part_id,
                        delta.content,
                    )

            if hasattr(msg, "tool_calls") and msg.tool_calls:
                abort = await handle_tool_calls(
                    msg,
                    formatter,
                    assistant_msg_id,
                    tool_call_parts,
                    handled_blocked_calls,
                )
                if abort:
                    stream_abort = True

        elif isinstance(msg, ToolMessage):
            await handle_tool_result(
                msg, formatter, assistant_msg_id, tool_call_parts, handled_blocked_calls
            )

    state.full_text = re.sub(r"</?think>", "", state.full_text)
    state.full_text = re.sub(
        r"^/(?:think|end_think)\s*\n?", "", state.full_text, flags=re.MULTILINE
    )

    return StreamResult(
        full_text=state.full_text,
        tool_call_parts=tool_call_parts,
        reasoning_part_id=state.reasoning_part_id,
        reasoning_start_time=state.reasoning_start_time,
        full_reasoning_text=state.full_reasoning_text,
        chunk_count=chunk_count,
    )

"""tool_handler — tool call initiation and result processing for agent streams."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.messages import AIMessageChunk, ToolMessage

    from backend.src.services.session.runner.event_formatter import EventFormatter


@dataclass
class ToolCallRecord:
    part_id: str
    tool_name: str
    tool_args: dict[str, Any]
    start: float


session_todos: dict[str, list[dict[str, Any]]] = {}


def normalize_tool_name(name: str) -> str:
    """Map LLM tool names to canonical lowercase names."""
    from backend.src.services.session.runner.constants import (
        normalize_tool_name as _normalize,
    )

    return _normalize(name)


async def handle_tool_calls(
    msg: AIMessageChunk,
    formatter: EventFormatter,
    assistant_msg_id: str,
    tool_call_parts: dict[str, ToolCallRecord],
    handled_blocked_calls: set[str],
) -> bool:
    from backend.src.services.session.runner.permission_handler import (
        handle_permission_request,
        handle_question_request,
    )
    from backend.src.storage.utils.converters import ulid_factory

    for tc in msg.tool_calls:
        call_id = None
        tool_name = None
        try:
            call_id = str(tc.get("id", "")) or ulid_factory()
            tool_name = tc.get("name", "unknown")
            tool_args = tc.get("args", {})
            part_id = ulid_factory()

            tool_call_parts[call_id] = ToolCallRecord(
                part_id=part_id,
                tool_name=tool_name,
                tool_args=tool_args,
                start=time.time(),
            )

            _ = await formatter.emit_tool_start(
                assistant_msg_id,
                call_id,
                tool_name,
                tool_args,
            )

            if tool_name == "Question":
                handled, abort = await handle_question_request(
                    formatter.emitter,
                    formatter.session_id,
                    assistant_msg_id,
                    call_id,
                    tool_args,
                )
                if handled:
                    handled_blocked_calls.add(call_id)
                    if abort:
                        return True
            else:
                handled, abort = await handle_permission_request(
                    formatter.emitter,
                    formatter.session_id,
                    assistant_msg_id,
                    call_id,
                    tool_name,
                    tool_args,
                )
                if handled:
                    handled_blocked_calls.add(call_id)
                    if abort:
                        return True
        except Exception as e:
            from loguru import logger as log

            log.error("error_processing_tool_call", error=str(e))
            raise

    return False


async def handle_tool_result(
    msg: ToolMessage,
    formatter: EventFormatter,
    assistant_msg_id: str,
    tool_call_parts: dict[str, ToolCallRecord],
    handled_blocked_calls: set[str],
) -> None:
    from backend.src.services.session.runner.utils import _parse_todo_output

    call_id = getattr(msg, "tool_call_id", None)

    if call_id and call_id in tool_call_parts and call_id not in handled_blocked_calls:
        tc_info = tool_call_parts[call_id]
        tool_name = tc_info.tool_name
        tool_args = tc_info.tool_args
        metadata: dict[str, Any] = {}
        if tool_name == "todowrite":
            todos = _parse_todo_output(str(msg.content))
            metadata = {"todos": todos}
            session_todos[formatter.session_id] = todos

        _ = await formatter.emit_tool_end(
            assistant_msg_id,
            call_id,
            tool_name,
            tool_args,
            str(msg.content),
            tc_info.start,
            metadata,
        )

        if tool_name == "todowrite":
            await formatter.emitter.emit_todo_updated(
                formatter.session_id, metadata.get("todos", [])
            )

"""
permission_handler — handles permission requests and questions via oc/state futures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backend.src.services.session.event_emitter import EventEmitter


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


def _normalize_tool_name(name: str) -> str:
    """map llm tool names to tui-expected lowercase names."""
    from backend.src.services.session.runner.constants import normalize_tool_name

    return normalize_tool_name(name)


async def handle_permission_request(
    emitter: EventEmitter,
    session_id: str,
    assistant_msg_id: str,
    call_id: str,
    tool_name: str,
    tool_args: dict[str, Any],
) -> tuple[bool, str]:
    """Handle a tool permission request. Returns (handled, stream_abort)."""
    from backend.src.api.routes.oc.state import ask_permission

    permission = _TOOL_PERMISSIONS.get(tool_name)
    if not permission:
        return False, ""

    request_id, future, request = ask_permission(
        session_id=session_id,
        permission=permission,
        patterns=[str(tool_args.get("path", ""))] if tool_args.get("path") else [""],
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
    await emitter.emit_permission_asked(request)
    reply = await future

    if reply == "reject":
        await emitter.emit_session_status(session_id, {"type": "idle"})
        return True, "abort"

    return True, ""


async def handle_question_request(
    emitter: EventEmitter,
    session_id: str,
    assistant_msg_id: str,
    call_id: str,
    tool_args: dict[str, Any],
) -> tuple[bool, str]:
    """Handle a Question tool request. Returns (handled, stream_abort)."""
    from backend.src.api.routes.oc.state import ask_question

    request_id, future, request = ask_question(
        session_id=session_id,
        questions=tool_args.get("questions", []),
        tool={
            "messageID": assistant_msg_id,
            "callID": call_id,
        },
    )
    await emitter.emit_question_asked(request)
    reply = await future

    if reply == "reject":
        await emitter.emit_session_status(session_id, {"type": "idle"})
        return True, "abort"

    return True, ""

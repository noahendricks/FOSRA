from __future__ import annotations

import asyncio
from typing import Any, cast

from loguru import logger

from backend.src.api.events import BusEvent, BusEventProperties, EventBus, event_bus
from backend.src.api.schemas.tui_event_schemas import TUIEvent

MAX_QUEUE_SIZE = 1000
MAX_RECENT_EVENTS = 100


class EventEmitter:
    def __init__(self, bus: EventBus | None = None) -> None:
        self._bus = bus or event_bus

    def subscribe(
        self, session_id: str | None = None
    ) -> tuple[str, asyncio.Queue[BusEvent]]:
        return self._bus.subscribe()

    def unsubscribe(self, sub_id: str) -> None:
        self._bus.unsubscribe(sub_id)

    def replay_missed(self, after_id: int):
        return self._bus.replay_missed(after_id)

    async def emit(self, event_type: str, properties: dict[str, Any]) -> None:
        tuiev = TUIEvent(type=event_type, properties=properties)
        await self._bus.publish(
            cast(
                BusEventProperties,
                cast(
                    object,
                    {
                        "type": tuiev.type,
                        "properties": tuiev.properties,
                    },
                ),
            )
        )

    async def emit_message_updated(self, info: dict[str, Any]) -> None:
        await self.emit("message.updated", {"info": info})

    async def emit_message_removed(self, session_id: str, message_id: str) -> None:
        await self.emit(
            "message.removed", {"sessionID": session_id, "messageID": message_id}
        )

    async def emit_message_part_updated(self, part: dict[str, Any]) -> None:
        await self.emit("message.part.updated", {"part": part})

    async def emit_message_part_delta(
        self,
        session_id: str,
        message_id: str,
        part_id: str,
        field: str,
        delta: str,
        part_type: str = "text",
    ) -> None:
        logger.bind(
            _structured={
                "session_id": session_id,
                "message_id": message_id,
                "part_id": part_id,
                "field": field,
                "delta_len": len(delta),
                "delta_preview": delta[:50] if delta else "",
            }
        ).debug("[EVENT EMITTER] emit_message_part_delta")
        await self.emit(
            "message.part.delta",
            {
                "sessionID": session_id,
                "messageID": message_id,
                "partID": part_id,
                "field": field,
                "delta": delta,
                "partType": part_type,
            },
        )

    async def emit_message_part_removed(
        self, session_id: str, message_id: str, part_id: str
    ) -> None:
        await self.emit(
            "message.part.removed",
            {
                "sessionID": session_id,
                "messageID": message_id,
                "partID": part_id,
            },
        )

    async def emit_session_created(self, info: dict[str, Any]) -> None:
        await self.emit("session.created", {"info": info})

    async def emit_session_updated(self, info: dict[str, Any]) -> None:
        await self.emit("session.updated", {"info": info})

    async def emit_session_deleted(self, info: dict[str, Any]) -> None:
        await self.emit("session.deleted", {"info": info})

    async def emit_session_status(
        self, session_id: str, status: dict[str, Any]
    ) -> None:
        await self.emit("session.status", {"sessionID": session_id, "status": status})

    async def emit_session_diff(self, session_id: str, diff: dict[str, Any]) -> None:
        await self.emit("session.diff", {"sessionID": session_id, "diff": diff})

    async def emit_session_error(
        self, session_id: str | None, error: dict[str, Any] | str
    ) -> None:
        if isinstance(error, str):
            error = {"message": error, "type": "unknown"}
        await self.emit("session.error", {"sessionID": session_id, "error": error})

    async def emit_permission_asked(self, permission: dict[str, Any]) -> None:
        await self.emit("permission.asked", permission)

    async def emit_permission_replied(
        self, session_id: str, request_id: str, reply: str
    ) -> None:
        await self.emit(
            "permission.replied",
            {
                "sessionID": session_id,
                "requestID": request_id,
                "reply": reply,
            },
        )

    async def emit_question_asked(self, question: dict[str, Any]) -> None:
        await self.emit("question.asked", question)

    async def emit_question_replied(
        self, session_id: str, request_id: str, answers: Any
    ) -> None:
        await self.emit(
            "question.replied",
            {
                "sessionID": session_id,
                "requestID": request_id,
                "answers": answers,
            },
        )

    async def emit_question_rejected(self, session_id: str, request_id: str) -> None:
        await self.emit(
            "question.rejected", {"sessionID": session_id, "requestID": request_id}
        )

    async def emit_todo_created(self, session_id: str, todo: dict[str, Any]) -> None:
        await self.emit("todo.created", {"sessionID": session_id, "todo": todo})

    async def emit_todo_updated(self, session_id: str, todo: dict[str, Any]) -> None:
        await self.emit("todo.updated", {"sessionID": session_id, "todo": todo})

    async def emit_todo_deleted(self, session_id: str, todo: dict[str, Any]) -> None:
        await self.emit("todo.deleted", {"sessionID": session_id, "todo": todo})

    async def emit_lsp_updated(self, status: dict[str, Any]) -> None:
        await self.emit("lsp.updated", status)

    async def emit_vcs_branch_updated(self, branch: dict[str, Any]) -> None:
        await self.emit("vcs.branch.updated", branch)

    async def emit_server_connected(self) -> None:
        await self.emit("server.connected", {})

    async def emit_server_instance_disposed(self, directory: str) -> None:
        await self.emit("server.instance.disposed", {"directory": directory})

    async def emit_installation_update_available(
        self, current_version: str, latest_version: str
    ) -> None:
        await self.emit(
            "installation.update-available",
            {"currentVersion": current_version, "latestVersion": latest_version},
        )

    async def emit_idle(self, session_id: str) -> None:
        await self.emit("idle", {"sessionID": session_id, "status": {"type": "idle"}})

    async def emit_busy(self, session_id: str) -> None:
        await self.emit("busy", {"sessionID": session_id, "status": {"type": "busy"}})


_global_emitter: EventEmitter | None = None


def get_event_emitter() -> EventEmitter:
    global _global_emitter
    if _global_emitter is None:
        _global_emitter = EventEmitter()
    return _global_emitter

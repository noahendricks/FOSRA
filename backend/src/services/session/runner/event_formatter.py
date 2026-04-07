"""
event_formatter — constructs and emits TUI-shaped events from agent stream data.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from backend.src.api.schemas.tui_schemas import (
    DEFAULT_MODEL_ID,
    DEFAULT_PROVIDER_ID,
    PROJECT_DIR,
    AssistantMessage,
    AssistantMessagePath,
    AssistantMessageTime,
    AssistantMessageTokens,
    AssistantMessageTokensCache,
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
    UserMessage,
    UserMessageModel,
    UserMessageTime,
)
from backend.src.storage.utils.converters import ulid_factory

if TYPE_CHECKING:
    from backend.src.services.session.event_emitter import EventEmitter


class EventFormatter:
    def __init__(
        self,
        emitter: EventEmitter,
        session_id: str,
        now: int,
        provider_id: str | None = None,
        model_id: str | None = None,
    ) -> None:
        self._emitter = emitter
        self._session_id = session_id
        self._now = now
        self._provider_id = provider_id
        self._model_id = model_id

    @property
    def emitter(self) -> EventEmitter:
        return self._emitter

    @property
    def session_id(self) -> str:
        return self._session_id

    async def emit_user_message(
        self,
        user_msg_id: str,
        agent_name: str,
        user_text: str,
    ) -> None:
        msg = UserMessage(
            id=user_msg_id,
            sessionID=self._session_id,
            role="user",
            time=UserMessageTime(created=self._now),
            agent=agent_name,
            model=UserMessageModel(
                providerID=self._provider_id or DEFAULT_PROVIDER_ID,
                modelID=self._model_id or DEFAULT_MODEL_ID,
            ),
        ).model_dump()
        await self._emitter.emit_message_updated(msg)

    async def emit_user_text_part(
        self,
        user_msg_id: str,
        text: str,
    ) -> None:
        part = TextPart(
            id=ulid_factory(),
            sessionID=self._session_id,
            messageID=user_msg_id,
            type="text",
            text=text,
            time=TextPartTime(start=self._now, end=self._now),
        ).model_dump()
        await self._emitter.emit_message_part_updated(part)

    async def emit_assistant_message_start(
        self,
        assistant_msg_id: str,
        user_msg_id: str,
        agent_name: str,
    ) -> tuple[str, str]:
        text_part_id = ulid_factory()
        step_start_id = ulid_factory()

        assistant_msg = AssistantMessage(
            id=assistant_msg_id,
            sessionID=self._session_id,
            role="assistant",
            time=AssistantMessageTime(created=self._now, completed=None),
            parentID=user_msg_id,
            modelID=self._model_id or DEFAULT_MODEL_ID,
            providerID=self._provider_id or DEFAULT_PROVIDER_ID,
            mode="default",
            agent=agent_name,
            path=AssistantMessagePath(cwd=PROJECT_DIR, root=PROJECT_DIR),
            cost=0,
            tokens=AssistantMessageTokens(
                input=0,
                output=0,
                reasoning=0,
                cache=AssistantMessageTokensCache(read=0, write=0),
            ),
        ).model_dump()
        await self._emitter.emit_message_updated(assistant_msg)

        step_start = StepStartPart(
            id=step_start_id,
            sessionID=self._session_id,
            messageID=assistant_msg_id,
            type="step-start",
        ).model_dump()
        await self._emitter.emit_message_part_updated(step_start)

        text_part = TextPart(
            id=text_part_id,
            sessionID=self._session_id,
            messageID=assistant_msg_id,
            type="text",
            text="",
            time=TextPartTime(start=self._now, end=None),
        ).model_dump()
        await self._emitter.emit_message_part_updated(text_part)

        return text_part_id, step_start_id

    async def emit_reasoning_start(
        self,
        assistant_msg_id: str,
        reasoning_part_id: str,
        reasoning_start_time: int,
    ) -> None:
        rp = ReasoningPart(
            id=reasoning_part_id,
            sessionID=self._session_id,
            messageID=assistant_msg_id,
            type="reasoning",
            text="",
            time=ReasoningPartTime(start=reasoning_start_time, end=None),
        ).model_dump()
        await self._emitter.emit_message_part_updated(rp)

    async def emit_text_delta(
        self,
        assistant_msg_id: str,
        part_id: str,
        text: str,
    ) -> None:
        await self._emitter.emit_message_part_delta(
            self._session_id,
            assistant_msg_id,
            part_id,
            "text",
            text,
        )

    async def emit_reasoning_delta(
        self,
        assistant_msg_id: str,
        reasoning_part_id: str,
        text: str,
    ) -> None:
        await self._emitter.emit_message_part_delta(
            self._session_id,
            assistant_msg_id,
            reasoning_part_id,
            "text",
            text,
            part_type="reasoning",
        )

    async def emit_tool_start(
        self,
        assistant_msg_id: str,
        call_id: str,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> str:
        part_id = ulid_factory()
        tool_part = ToolPart(
            id=part_id,
            sessionID=self._session_id,
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
        await self._emitter.emit_message_part_updated(tool_part)
        return part_id

    async def emit_tool_end(
        self,
        assistant_msg_id: str,
        call_id: str,
        tool_name: str,
        tool_args: dict[str, Any],
        output: str,
        start_time: float,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        new_text_part_id = ulid_factory()
        end_time = int(time.time())
        tool_part = ToolPart(
            id=new_text_part_id,
            sessionID=self._session_id,
            messageID=assistant_msg_id,
            type="tool",
            callID=call_id,
            tool=tool_name,
            state=ToolStateCompleted(
                status="completed",
                input=tool_args,
                output=output[:2000],
                title=tool_name,
                metadata=metadata or {},
                time=ToolStateCompletedTime(start=int(start_time), end=end_time),
            ),
        ).model_dump()
        await self._emitter.emit_message_part_updated(tool_part)
        return new_text_part_id

    async def emit_step_finish(
        self,
        assistant_msg_id: str,
        step_finish_id: str,
        reason: str = "stop",
    ) -> None:
        part = StepFinishPart(
            id=step_finish_id,
            sessionID=self._session_id,
            messageID=assistant_msg_id,
            type="step-finish",
            reason=reason,
            cost=0,
            tokens=StepFinishPartTokens(
                input=0,
                output=0,
                reasoning=0,
                cache=StepFinishPartTokensCache(read=0, write=0),
            ),
        ).model_dump()
        await self._emitter.emit_message_part_updated(part)

    async def emit_text_final(
        self,
        assistant_msg_id: str,
        text_part_id: str,
        full_text: str,
        end_time: int,
    ) -> None:
        part = TextPart(
            id=text_part_id,
            sessionID=self._session_id,
            messageID=assistant_msg_id,
            type="text",
            text=full_text,
            time=TextPartTime(start=self._now, end=end_time),
        ).model_dump()
        await self._emitter.emit_message_part_updated(part)

    async def emit_reasoning_final(
        self,
        assistant_msg_id: str,
        reasoning_part_id: str,
        reasoning_start_time: int,
        end_time: int,
        full_reasoning_text: str = "",
    ) -> None:
        part = ReasoningPart(
            id=reasoning_part_id,
            sessionID=self._session_id,
            messageID=assistant_msg_id,
            type="reasoning",
            text=full_reasoning_text,
            time=ReasoningPartTime(start=reasoning_start_time, end=end_time),
        ).model_dump()
        await self._emitter.emit_message_part_updated(part)

    async def emit_assistant_final(
        self,
        assistant_msg_id: str,
        user_msg_id: str,
        agent_name: str,
        end_time: int,
        finish: str = "stop",
        error: UnknownError | None = None,
    ) -> None:
        msg = AssistantMessage(
            id=assistant_msg_id,
            sessionID=self._session_id,
            role="assistant",
            time=AssistantMessageTime(created=self._now, completed=end_time),
            parentID=user_msg_id,
            modelID=self._model_id or DEFAULT_MODEL_ID,
            providerID=self._provider_id or DEFAULT_PROVIDER_ID,
            mode="default",
            agent=agent_name,
            path=AssistantMessagePath(cwd=PROJECT_DIR, root=PROJECT_DIR),
            cost=0,
            tokens=AssistantMessageTokens(
                input=0,
                output=0,
                reasoning=0,
                cache=AssistantMessageTokensCache(read=0, write=0),
            ),
            finish=finish,
            error=error,
        ).model_dump()
        await self._emitter.emit_message_updated(msg)

    async def emit_session_error(self, error: dict[str, Any] | str) -> None:
        await self._emitter.emit_session_error(self._session_id, error)

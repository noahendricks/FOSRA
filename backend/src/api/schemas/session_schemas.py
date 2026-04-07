"""session schemas — Session, SessionStatus, related types."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel


class SessionTime(BaseModel):
    created: int
    updated: int
    compacting: int | None = None
    archived: int | None = None


class SessionRevert(BaseModel):
    message_id: str
    part_id: str | None = None
    snapshot: str | None = None
    diff: str | None = None


class SessionStatusIdle(BaseModel):
    type: Literal["idle"]


class SessionStatusBusy(BaseModel):
    type: Literal["busy"]


class SessionStatusRetry(BaseModel):
    type: Literal["retry"]
    attempt: int
    message: str
    next: int


SessionStatus = SessionStatusIdle | SessionStatusBusy | SessionStatusRetry


class SessionModelCostCache(BaseModel):
    read: int | float
    write: int | float


class SessionModelCost(BaseModel):
    input: int | float
    output: int | float
    cache: SessionModelCostCache | None = None


class SessionModelLimit(BaseModel):
    context: int
    input: int | float | None = None
    output: int | float | None = None


class SessionModelInfo(BaseModel):
    provider_id: str
    model_id: str
    cost: SessionModelCost | None = None
    limit: SessionModelLimit | None = None


class SessionMetadataModel(BaseModel):
    model: SessionModelInfo | None = None
    agent: str | None = None


class Session(BaseModel):
    id: str
    user_id: str | None = None
    directory: str
    parent_id: str | None = None
    title: str
    version: str
    time: SessionTime
    permission: Any | None = None
    revert: SessionRevert | None = None
    metadata: SessionMetadataModel | None = None


class GlobalSession(BaseModel):
    id: str
    user_id: str | None = None
    directory: str
    parent_id: str | None = None
    title: str
    version: str
    time: SessionTime
    permission: Any | None = None
    revert: Optional[SessionRevert] = None

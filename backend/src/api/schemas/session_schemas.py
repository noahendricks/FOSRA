"""session schemas — Session, SessionStatus, related types."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel


class SessionTime(BaseModel):
    created: int
    updated: int
    compacting: int | None = None
    archived: int | None = None


class SessionSummary(BaseModel):
    additions: int
    deletions: int
    files: int
    diffs: list[Any] | None = None  # FileDiff — avoid circular import


class SessionShare(BaseModel):
    url: str


class SessionRevert(BaseModel):
    messageID: str
    partID: str | None = None
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
    read: int
    write: int


class SessionModelCost(BaseModel):
    input: int
    output: int
    cache: SessionModelCostCache | None = None


class SessionModelLimit(BaseModel):
    context: int
    input: int | None = None
    output: int | None = None


class SessionModelInfo(BaseModel):
    providerID: str
    modelID: str
    cost: SessionModelCost | None = None
    limit: SessionModelLimit | None = None


class SessionMetadataModel(BaseModel):
    model: SessionModelInfo | None = None
    agent: str | None = None


class Session(BaseModel):
    id: str
    slug: str
    projectID: str
    workspaceID: str | None = None
    directory: str
    parentID: str | None = None
    summary: Optional[SessionSummary] = None
    share: Optional[SessionShare] = None
    title: str
    version: str
    time: SessionTime
    permission: Any | None = None
    revert: Optional[SessionRevert] = None
    metadata: SessionMetadataModel | None = None


class GlobalSession(BaseModel):
    id: str
    slug: str
    projectID: str
    workspaceID: str | None = None
    directory: str
    parentID: str | None = None
    summary: Optional[SessionSummary] = None
    share: Optional[SessionShare] = None
    title: str
    version: str
    time: SessionTime
    permission: Any | None = None
    revert: Optional[SessionRevert] = None
    project: Any | None = None

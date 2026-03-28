"""session schemas — Session, SessionStatus, related types."""

from __future__ import annotations

from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel


class SessionTime(BaseModel):
    created: int
    updated: int
    compacting: Optional[int] = None
    archived: Optional[int] = None


class SessionSummary(BaseModel):
    additions: int
    deletions: int
    files: int
    diffs: Optional[List[Any]] = None  # FileDiff — avoid circular import


class SessionShare(BaseModel):
    url: str


class SessionRevert(BaseModel):
    messageID: str
    partID: Optional[str] = None
    snapshot: Optional[str] = None
    diff: Optional[str] = None


class SessionStatusIdle(BaseModel):
    type: Literal["idle"]


class SessionStatusBusy(BaseModel):
    type: Literal["busy"]


class SessionStatusRetry(BaseModel):
    type: Literal["retry"]
    attempt: int
    message: str
    next: int


SessionStatus = Union[SessionStatusIdle, SessionStatusBusy, SessionStatusRetry]


class Session(BaseModel):
    id: str
    slug: str
    projectID: str
    workspaceID: Optional[str] = None
    directory: str
    parentID: Optional[str] = None
    summary: Optional[SessionSummary] = None
    share: Optional[SessionShare] = None
    title: str
    version: str
    time: SessionTime
    permission: Optional[Any] = None
    revert: Optional[SessionRevert] = None


class GlobalSession(BaseModel):
    id: str
    slug: str
    projectID: str
    workspaceID: Optional[str] = None
    directory: str
    parentID: Optional[str] = None
    summary: Optional[SessionSummary] = None
    share: Optional[SessionShare] = None
    title: str
    version: str
    time: SessionTime
    permission: Optional[Any] = None
    revert: Optional[SessionRevert] = None
    project: Optional[Any] = None

"""workspace schemas — Workspace, Project, Worktree, Auth, Symbol, error responses, Pty."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel


# ---- PROJECT ----


class ProjectIcon(BaseModel):
    url: Optional[str] = None
    override: Optional[str] = None
    color: Optional[str] = None


class ProjectCommands(BaseModel):
    start: Optional[str] = None


class ProjectTime(BaseModel):
    created: int
    updated: int
    initialized: Optional[int] = None


class Project(BaseModel):
    id: str
    worktree: str
    vcs: Optional[Literal["git"]] = None
    name: Optional[str] = None
    icon: Optional[ProjectIcon] = None
    commands: Optional[ProjectCommands] = None
    time: ProjectTime
    sandboxes: List[str]


class ProjectSummary(BaseModel):
    id: str
    name: Optional[str] = None
    worktree: str


# ---- WORKSPACE ----


class Workspace(BaseModel):
    id: str
    type: str
    branch: Optional[str] = None
    name: Optional[str] = None
    directory: Optional[str] = None
    extra: Optional[Any] = None
    projectID: str


# ---- WORKTREE ----


class Worktree(BaseModel):
    name: str
    branch: str
    directory: str


class WorktreeCreateInput(BaseModel):
    name: Optional[str] = None
    startCommand: Optional[str] = None


class WorktreeRemoveInput(BaseModel):
    directory: str


class WorktreeResetInput(BaseModel):
    directory: str


# ---- SYMBOL / CODE NAVIGATION ----


class RangeStart(BaseModel):
    line: int
    character: int


class RangeEnd(BaseModel):
    line: int
    character: int


class Range(BaseModel):
    start: RangeStart
    end: RangeEnd


class SymbolLocation(BaseModel):
    uri: str
    range: Range


class Symbol(BaseModel):
    name: str
    kind: int
    location: SymbolLocation


# ---- AUTH ----


class OAuth(BaseModel):
    type: Literal["oauth"]
    refresh: str
    access: str
    expires: int
    accountId: Optional[str] = None
    enterpriseUrl: Optional[str] = None


class ApiAuth(BaseModel):
    type: Literal["api"]
    key: str


class WellKnownAuth(BaseModel):
    type: Literal["wellknown"]
    key: str
    token: str


class Auth(BaseModel):
    pass


# ---- ERROR RESPONSES ----


class NotFoundErrorData(BaseModel):
    message: str


class NotFoundError(BaseModel):
    name: Literal["NotFoundError"]
    data: NotFoundErrorData


class BadRequestError(BaseModel):
    data: Any
    errors: List[Dict[str, Any]]
    success: Literal[False]


# ---- PTY ----


class Pty(BaseModel):
    id: str
    title: str
    command: str
    args: List[str]
    cwd: str
    status: Literal["running", "exited"]
    pid: int

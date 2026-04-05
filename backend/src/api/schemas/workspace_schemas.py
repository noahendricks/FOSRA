"""workspace schemas — Workspace, Project, Worktree, Auth, Symbol, error responses, Pty."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


# ---- PROJECT ----


class ProjectIcon(BaseModel):
    url: str | None = None
    override: str | None = None
    color: str | None = None


class ProjectCommands(BaseModel):
    start: str | None = None


class ProjectTime(BaseModel):
    created: int
    updated: int
    initialized: int | None = None


class Project(BaseModel):
    id: str
    worktree: str
    vcs: Literal["git"] | None = None
    name: str | None = None
    icon: ProjectIcon | None = None
    commands: ProjectCommands | None = None
    time: ProjectTime
    sandboxes: list[str]


class ProjectSummary(BaseModel):
    id: str
    name: str | None = None
    worktree: str


# ---- WORKSPACE ----


class Workspace(BaseModel):
    id: str
    type: str
    branch: str | None = None
    name: str | None = None
    directory: str | None = None
    extra: Any | None = None
    projectID: str


# ---- WORKTREE ----


class Worktree(BaseModel):
    name: str
    branch: str
    directory: str


class WorktreeCreateInput(BaseModel):
    name: str | None = None
    startCommand: str | None = None


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
    accountId: str | None = None
    enterpriseUrl: str | None = None


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
    errors: list[dict[str, Any]]
    success: Literal[False]


# ---- PTY ----


class Pty(BaseModel):
    id: str
    title: str
    command: str
    args: list[str]
    cwd: str
    status: Literal["running", "exited"]
    pid: int

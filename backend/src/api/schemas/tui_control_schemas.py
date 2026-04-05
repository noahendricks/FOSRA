"""tui control schemas — PromptRequest, FileDiff, Permission, Question, Todo, Path, Command, etc."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.src.api.schemas.base import BaseModelFlex


# ---- LEGACY WORKSPACE MESSAGE TYPES ----


class TextPart(BaseModelFlex):
    type: str
    text: str


class FilePart(BaseModelFlex):
    type: str
    name: str
    size: int
    filename: str
    bytes: bytes
    media_type: str
    url: str | None = None


UIMessagePart = TextPart | FilePart


class UIMessage(BaseModelFlex):
    id: str
    role: str
    parts: list[UIMessagePart]
    message_metadata: dict[str, Any] | None = None
    sources: list[dict[str, Any]] | None = None


# ---- FILE DIFF ----


class FileDiff(BaseModel):
    file: str
    before: str
    after: str
    additions: int
    deletions: int
    status: Literal["added", "deleted", "modified"] | None = None


class RangeStart(BaseModel):
    line: int
    character: int


class RangeEnd(BaseModel):
    line: int
    character: int


class Range(BaseModel):
    start: RangeStart
    end: RangeEnd


class FilePartSourceText(BaseModel):
    value: str
    start: int
    end: int


class FilePartSource(BaseModel):
    text: FilePartSourceText
    type: Literal["file"]
    path: str


class SymbolSource(BaseModel):
    text: FilePartSourceText
    type: Literal["symbol"]
    path: str
    range: Range
    name: str
    kind: int


class ResourceSource(BaseModel):
    text: FilePartSourceText
    type: Literal["resource"]
    clientName: str
    uri: str


# ---- PERMISSION ----


class PermissionRequestTool(BaseModel):
    messageID: str
    callID: str


class PermissionRequest(BaseModel):
    id: str
    sessionID: str
    permission: str
    patterns: list[str]
    metadata: dict[str, Any]
    always: list[str]
    tool: PermissionRequestTool | None = None


class PermissionAction(BaseModel):
    reply: Literal["once", "always", "reject"]
    message: str = ""


class PermissionRule(BaseModel):
    permission: str
    pattern: str
    action: PermissionAction


class PermissionRuleset(BaseModel):
    pass


# ---- QUESTION ----


class QuestionOption(BaseModel):
    label: str
    description: str


class QuestionInfo(BaseModel):
    question: str
    header: str
    options: list[QuestionOption]
    multiple: bool | None = None
    custom: bool | None = None


class QuestionRequestTool(BaseModel):
    messageID: str
    callID: str


class QuestionRequest(BaseModel):
    id: str
    sessionID: str
    questions: list[QuestionInfo]
    tool: QuestionRequestTool | None = None


class QuestionAnswer(BaseModel):
    sessionID: str
    answers: list[list[str]]


class QuestionReject(BaseModel):
    sessionID: str


# ---- TODO ----


class Todo(BaseModel):
    content: str
    status: str
    priority: str


# ---- PATH / COMMAND ----


class Path(BaseModel):
    home: str
    state: str
    config: str
    worktree: str
    directory: str


class Command(BaseModel):
    name: str
    description: str | None = None
    agent: str | None = None
    model: str | None = None
    source: Literal["command", "mcp", "skill"] | None = None
    template: str
    subtask: bool | None = None
    hints: list[str] = Field(default_factory=list)


# ---- PROMPT REQUEST ----


class SubtaskPartModel(BaseModel):
    providerID: str
    modelID: str


class TextPartTime(BaseModel):
    start: int
    end: int | None = None


class TextPartInput(BaseModel):
    id: str | None = None
    type: Literal["text"]
    text: str
    synthetic: bool | None = None
    ignored: bool | None = None
    time: TextPartTime | None = None
    metadata: dict[str, Any] | None = None


class FilePartInput(BaseModel):
    id: str | None = None
    type: Literal["file"]
    mime: str
    filename: str | None = None
    url: str
    source: FilePartSource | None = None


class AgentPartInputSource(BaseModel):
    value: str
    start: int
    end: int


class AgentPartInput(BaseModel):
    id: str | None = None
    type: Literal["agent"]
    name: str
    source: AgentPartInputSource | None = None


class SubtaskPartInput(BaseModel):
    id: str | None = None
    type: Literal["subtask"]
    prompt: str
    description: str
    agent: str
    model: SubtaskPartModel | None = None
    command: str | None = None


class AgentModel(BaseModel):
    modelID: str
    providerID: str


class PromptRequest(BaseModel):
    sessionID: str | None = None
    parts: list[TextPartInput | FilePartInput | AgentPartInput | SubtaskPartInput] = (
        Field(default_factory=list)
    )
    model: AgentModel | None = None
    agent: str | None = None
    variant: str | None = None
    providerID: str | None = None
    modelID: str | None = None

    model_config = {"extra": "allow"}

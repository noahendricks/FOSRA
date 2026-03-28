"""tui control schemas — PromptRequest, FileDiff, Permission, Question, Todo, Path, Command, etc."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class FileDiff(BaseModel):
    file: str
    before: str
    after: str
    additions: int
    deletions: int
    status: Optional[Literal["added", "deleted", "modified"]] = None


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
    patterns: List[str]
    metadata: Dict[str, Any]
    always: List[str]
    tool: Optional[PermissionRequestTool] = None


class PermissionAction(BaseModel):
    pass


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
    options: List[QuestionOption]
    multiple: Optional[bool] = None
    custom: Optional[bool] = None


class QuestionRequestTool(BaseModel):
    messageID: str
    callID: str


class QuestionRequest(BaseModel):
    id: str
    sessionID: str
    questions: List[QuestionInfo]
    tool: Optional[QuestionRequestTool] = None


class QuestionAnswer(BaseModel):
    pass


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
    description: Optional[str] = None
    agent: Optional[str] = None
    model: Optional[str] = None
    source: Optional[Literal["command", "mcp", "skill"]] = None
    template: str
    subtask: Optional[bool] = None
    hints: List[str] = Field(default_factory=list)


# ---- PROMPT REQUEST ----


class SubtaskPartModel(BaseModel):
    providerID: str
    modelID: str


class TextPartTime(BaseModel):
    start: int
    end: Optional[int] = None


class TextPartInput(BaseModel):
    id: Optional[str] = None
    type: Literal["text"]
    text: str
    synthetic: Optional[bool] = None
    ignored: Optional[bool] = None
    time: Optional[TextPartTime] = None
    metadata: Optional[Dict[str, Any]] = None


class FilePartInput(BaseModel):
    id: Optional[str] = None
    type: Literal["file"]
    mime: str
    filename: Optional[str] = None
    url: str
    source: Optional[FilePartSource] = None


class AgentPartInputSource(BaseModel):
    value: str
    start: int
    end: int


class AgentPartInput(BaseModel):
    id: Optional[str] = None
    type: Literal["agent"]
    name: str
    source: Optional[AgentPartInputSource] = None


class SubtaskPartInput(BaseModel):
    id: Optional[str] = None
    type: Literal["subtask"]
    prompt: str
    description: str
    agent: str
    model: Optional[SubtaskPartModel] = None
    command: Optional[str] = None


class AgentModel(BaseModel):
    modelID: str
    providerID: str


class PromptRequest(BaseModel):
    sessionID: Optional[str] = None
    messageID: Optional[str] = None
    parts: List[
        Union[TextPartInput, FilePartInput, AgentPartInput, SubtaskPartInput]
    ] = Field(default_factory=list)
    model: Optional[AgentModel] = None
    agent: Optional[str] = None
    variant: Optional[str] = None
    providerID: Optional[str] = None
    modelID: Optional[str] = None

    model_config = {"extra": "allow"}

"""message schemas — UserMessage, AssistantMessage, all Part types."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    pass

Message = Union["UserMessage", "AssistantMessage"]
Part = Union[
    "TextPart",
    "ToolPart",
    "ReasoningPart",
    "FilePart",
    "StepStartPart",
    "StepFinishPart",
    "AgentPart",
    "SubtaskPart",
    "RetryPart",
    "CompactionPart",
    "SnapshotPart",
    "PatchPart",
]
ToolState = Union[
    "ToolStatePending", "ToolStateRunning", "ToolStateCompleted", "ToolStateError"
]


class OutputFormat(BaseModel):
    type: Literal["text"]


# ---- USER MESSAGE ----


class UserMessageTime(BaseModel):
    created: int


class UserMessageModel(BaseModel):
    providerID: str
    modelID: str


class UserMessageSummary(BaseModel):
    title: Optional[str] = None
    body: Optional[str] = None
    diffs: List[Any] = Field(default_factory=list)  # FileDiff — avoid circular


class UserMessage(BaseModel):
    id: str
    sessionID: str
    role: Literal["user"]
    time: UserMessageTime
    format: Optional[OutputFormat] = None
    summary: Optional[UserMessageSummary] = None
    agent: str
    model: UserMessageModel
    system: Optional[str] = None
    tools: Optional[Dict[str, bool]] = None
    variant: Optional[str] = None


# ---- ASSISTANT MESSAGE ----


class AssistantMessageTime(BaseModel):
    created: int
    completed: Optional[int] = None


class AssistantMessagePath(BaseModel):
    cwd: str
    root: str


class AssistantMessageTokensCache(BaseModel):
    read: int
    write: int


class AssistantMessageTokens(BaseModel):
    total: Optional[int] = None
    input: int
    output: int
    reasoning: int
    cache: AssistantMessageTokensCache


class AssistantMessage(BaseModel):
    id: str
    sessionID: str
    role: Literal["assistant"]
    time: AssistantMessageTime
    error: Optional[Any] = None  # ApiError — avoid circular
    parentID: str
    modelID: str
    providerID: str
    mode: str
    agent: str
    path: AssistantMessagePath
    summary: Optional[bool] = None
    cost: float
    tokens: AssistantMessageTokens
    structured: Optional[Any] = None
    variant: Optional[str] = None
    finish: Optional[str] = None


# ---- PART TYPES ----


class TextPartTime(BaseModel):
    start: int
    end: Optional[int] = None


class TextPart(BaseModel):
    id: str
    sessionID: str
    messageID: str
    type: Literal["text"]
    text: str
    synthetic: Optional[bool] = None
    ignored: Optional[bool] = None
    time: Optional[TextPartTime] = None
    metadata: Optional[Dict[str, Any]] = None


class SubtaskPartModel(BaseModel):
    providerID: str
    modelID: str


class SubtaskPart(BaseModel):
    id: str
    sessionID: str
    messageID: str
    type: Literal["subtask"]
    prompt: str
    description: str
    agent: str
    model: Optional[SubtaskPartModel] = None
    command: Optional[str] = None


class ReasoningPartTime(BaseModel):
    start: int
    end: Optional[int] = None


class ReasoningPart(BaseModel):
    id: str
    sessionID: str
    messageID: str
    type: Literal["reasoning"]
    text: str
    metadata: Optional[Dict[str, Any]] = None
    time: ReasoningPartTime


class FilePartSourceText(BaseModel):
    value: str
    start: int
    end: int


class FilePartSource(BaseModel):
    text: FilePartSourceText
    type: Literal["file"]
    path: str


class FilePart(BaseModel):
    id: str
    sessionID: str
    messageID: str
    type: Literal["file"]
    mime: str
    filename: Optional[str] = None
    url: str
    source: Optional[FilePartSource] = None


class StepStartPart(BaseModel):
    id: str
    sessionID: str
    messageID: str
    type: Literal["step-start"]
    snapshot: Optional[str] = None


class StepFinishPartTokensCache(BaseModel):
    read: int
    write: int


class StepFinishPartTokens(BaseModel):
    total: Optional[int] = None
    input: int
    output: int
    reasoning: int
    cache: StepFinishPartTokensCache


class StepFinishPart(BaseModel):
    id: str
    sessionID: str
    messageID: str
    type: Literal["step-finish"]
    reason: str
    snapshot: Optional[str] = None
    cost: float
    tokens: StepFinishPartTokens


class SnapshotPart(BaseModel):
    id: str
    sessionID: str
    messageID: str
    type: Literal["snapshot"]
    snapshot: str


class PatchPart(BaseModel):
    id: str
    sessionID: str
    messageID: str
    type: Literal["patch"]
    hash: str
    files: List[str]


class AgentPartSource(BaseModel):
    value: str
    start: int
    end: int


class AgentPart(BaseModel):
    id: str
    sessionID: str
    messageID: str
    type: Literal["agent"]
    name: str
    source: Optional[AgentPartSource] = None


class RetryPartTime(BaseModel):
    created: int


class RetryPart(BaseModel):
    id: str
    sessionID: str
    messageID: str
    type: Literal["retry"]
    attempt: int
    error: Optional[Any] = None  # ApiError — avoid circular
    time: RetryPartTime


class CompactionPart(BaseModel):
    id: str
    sessionID: str
    messageID: str
    type: Literal["compaction"]
    auto: bool
    overflow: Optional[bool] = None


# ---- TOOL STATE ----


class ToolStatePending(BaseModel):
    status: Literal["pending"]
    input: Dict[str, Any]
    raw: str


class ToolStateRunningTime(BaseModel):
    start: int


class ToolStateRunning(BaseModel):
    status: Literal["running"]
    input: Dict[str, Any]
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    time: ToolStateRunningTime


class ToolStateCompletedTime(BaseModel):
    start: int
    end: int
    compacted: Optional[int] = None


class ToolStateCompleted(BaseModel):
    status: Literal["completed"]
    input: Dict[str, Any]
    output: str
    title: str
    metadata: Dict[str, Any]
    time: ToolStateCompletedTime
    attachments: Optional[List[FilePart]] = None


class ToolStateErrorTime(BaseModel):
    start: int
    end: int


class ToolStateError(BaseModel):
    status: Literal["error"]
    input: Dict[str, Any]
    error: str
    metadata: Optional[Dict[str, Any]] = None
    time: ToolStateErrorTime


class ToolPart(BaseModel):
    id: str
    sessionID: str
    messageID: str
    type: Literal["tool"]
    callID: str
    tool: str
    state: ToolState
    metadata: Optional[Dict[str, Any]] = None

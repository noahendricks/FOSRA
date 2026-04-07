"""message schemas — UserMessage, AssistantMessage, all Part types."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal, Union

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    pass


class OutputFormat(BaseModel):
    type: Literal["text"]


# ---- USER MESSAGE ----


class UserMessageTime(BaseModel):
    created: int


class UserMessageModel(BaseModel):
    providerID: str
    modelID: str


class UserMessageSummary(BaseModel):
    title: str | None = None
    body: str | None = None
    diffs: list[Any] = Field(default_factory=list)  # FileDiff — avoid circular


class UserMessage(BaseModel):
    id: str
    sessionID: str
    role: Literal["user"]
    time: UserMessageTime
    format: OutputFormat | None = None
    summary: UserMessageSummary | None = None
    agent: str
    model: UserMessageModel
    system: str | None = None
    tools: dict[str, bool] | None = None
    variant: str | None = None


# ---- ASSISTANT MESSAGE ----


class AssistantMessageTime(BaseModel):
    created: int
    completed: int | None = None


class AssistantMessagePath(BaseModel):
    cwd: str
    root: str


class AssistantMessageTokensCache(BaseModel):
    read: int
    write: int


class AssistantMessageTokens(BaseModel):
    total: int | None = None
    input: int
    output: int
    reasoning: int
    cache: AssistantMessageTokensCache


class AssistantMessage(BaseModel):
    id: str
    sessionID: str
    role: Literal["assistant"]
    time: AssistantMessageTime
    error: Any | None = None  # ApiError — avoid circular
    parentID: str
    modelID: str
    providerID: str
    mode: str
    agent: str
    path: AssistantMessagePath
    summary: bool | None = None
    cost: float
    tokens: AssistantMessageTokens
    structured: Any | None = None
    variant: str | None = None
    finish: str | None = None
    parts: list[Part] | None = None


# ---- PART TYPES ----


class TextPartTime(BaseModel):
    start: int
    end: int | None = None


class TextPart(BaseModel):
    id: str
    sessionID: str
    messageID: str
    type: Literal["text"]
    text: str
    synthetic: bool | None = None
    ignored: bool | None = None
    time: TextPartTime | None = None
    metadata: dict[str, Any] | None = None


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
    model: SubtaskPartModel | None = None
    command: str | None = None


class ReasoningPartTime(BaseModel):
    start: int
    end: int | None = None


class ReasoningPart(BaseModel):
    id: str
    sessionID: str
    messageID: str
    type: Literal["reasoning"]
    text: str
    metadata: dict[str, Any] | None = None
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
    filename: str | None = None
    url: str
    source: FilePartSource | None = None


class StepStartPart(BaseModel):
    id: str
    sessionID: str
    messageID: str
    type: Literal["step-start"]
    snapshot: str | None = None


class StepFinishPartTokensCache(BaseModel):
    read: int
    write: int


class StepFinishPartTokens(BaseModel):
    total: int | None = None
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
    snapshot: str | None = None
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
    files: list[str]


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
    source: AgentPartSource | None = None


class RetryPartTime(BaseModel):
    created: int


class RetryPart(BaseModel):
    id: str
    sessionID: str
    messageID: str
    type: Literal["retry"]
    attempt: int
    error: Any | None = None  # ApiError — avoid circular
    time: RetryPartTime


class CompactionPart(BaseModel):
    id: str
    sessionID: str
    messageID: str
    type: Literal["compaction"]
    auto: bool
    overflow: bool | None = None


# ---- TOOL STATE ----


class ToolStatePending(BaseModel):
    status: Literal["pending"]
    input: dict[str, Any]
    raw: str


class ToolStateRunningTime(BaseModel):
    start: int


class ToolStateRunning(BaseModel):
    status: Literal["running"]
    input: dict[str, Any]
    title: str | None = None
    metadata: dict[str, Any] | None = None
    time: ToolStateRunningTime


class ToolStateCompletedTime(BaseModel):
    start: int
    end: int
    compacted: int | None = None


class ToolStateCompleted(BaseModel):
    status: Literal["completed"]
    input: dict[str, Any]
    output: str
    title: str
    metadata: dict[str, Any]
    time: ToolStateCompletedTime
    attachments: list[FilePart] | None = None


class ToolStateErrorTime(BaseModel):
    start: int
    end: int


class ToolStateError(BaseModel):
    status: Literal["error"]
    input: dict[str, Any]
    error: str
    metadata: dict[str, Any] | None = None
    time: ToolStateErrorTime


ToolState = Annotated[
    Union[ToolStatePending, ToolStateRunning, ToolStateCompleted, ToolStateError],
    Field(discriminator="status"),
]



class ToolPart(BaseModel):
    id: str
    sessionID: str
    messageID: str
    type: Literal["tool"]
    callID: str
    tool: str
    state: ToolState
    metadata: dict[str, Any] | None = None


Message = UserMessage | AssistantMessage

Part = (
    TextPart
    | SubtaskPart
    | ReasoningPart
    | FilePart
    | StepStartPart
    | StepFinishPart
    | SnapshotPart
    | PatchPart
    | AgentPart
    | RetryPart
    | ToolPart
    | CompactionPart
)

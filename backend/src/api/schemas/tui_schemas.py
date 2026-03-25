"""
tui-compatible types and fosra → tui shape transformers.

maps fosra domain objects (convos, messages) into the shapes the
solidjs tui expects for sessions, messages, parts, and events.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from backend.src.storage.utils.converters import ulid_factory

PROJECT_DIR = os.environ.get("FOSRA_PROJECT_DIR", os.getcwd())
DEFAULT_USER_ID = os.environ.get("FOSRA_USER_ID", "dev-user000")
DEFAULT_PROJECT_ID = "default"
DEFAULT_VERSION = "2"
DEFAULT_PROVIDER_ID = "litellm"
DEFAULT_MODEL_ID = "default"


def _ts(dt: datetime | None) -> float:
    """datetime → unix timestamp (seconds)."""
    return dt.timestamp() if dt else time.time()


# =============================================================================
# TRANSFORMERS
# =============================================================================


def convo_to_session(
    convo_id: str,
    user_id: str,
    title: str | None = None,
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
    message_count: int = 0,
) -> Session:
    """convert fosra convo fields into a tui Session."""
    now = time.time()
    return Session(
        id=convo_id,
        slug=convo_id[:8],
        projectID=DEFAULT_PROJECT_ID,
        workspaceID=None,
        directory=PROJECT_DIR,
        parentID=None,
        title=title or "New Convo",
        version=DEFAULT_VERSION,
        time=SessionTime(
            created=int(_ts(created_at) if created_at else now),
            updated=int(_ts(updated_at) if updated_at else now),
        ),
    )


def convo_list_item_to_session(item: ConvoListItemResponse) -> Session:
    """convert a ConvoListItemResponse to a tui Session."""
    return convo_to_session(
        convo_id=item.convo_id,
        user_id=item.user_id,
        title=item.title,
        created_at=item.created_at,
        updated_at=item.updated_at,
        message_count=item.message_count,
    )


def convo_full_to_session(convo: ConvoFullResponse) -> Session:
    """convert a ConvoFullResponse to a tui Session."""
    return convo_to_session(
        convo_id=convo.convo_id,
        user_id=convo.user_id,
        title=convo.title,
        created_at=convo.created_at,
        updated_at=convo.updated_at,
        message_count=convo.message_count,
    )


def message_to_tui(msg: MessageResponse, session_id: str) -> dict[str, Any]:
    """
    convert a MessageResponse into a tui message dict.

    returns: {"info": Message, "parts": Part[]}
    """
    msg_id = msg.message_id or ulid_factory()
    created = int(_ts(msg.timestamp) if msg.timestamp else time.time())

    if msg.role.value == "user":
        info = UserMessage(
            id=msg_id,
            sessionID=session_id,
            role="user",
            time=UserMessageTime(created=created),
            agent="fosra",
            model=UserMessageModel(
                providerID=DEFAULT_PROVIDER_ID, modelID=DEFAULT_MODEL_ID
            ),
        )
    else:
        info = AssistantMessage(
            id=msg_id,
            sessionID=session_id,
            role="assistant",
            time=AssistantMessageTime(created=created, completed=created),
            parentID=msg.parent_id or "",
            modelID=DEFAULT_MODEL_ID,
            providerID=DEFAULT_PROVIDER_ID,
            mode="default",
            agent="fosra",
            path=AssistantMessagePath(cwd=PROJECT_DIR, root=PROJECT_DIR),
            cost=1.0,
            tokens=AssistantMessageTokens(
                input=1,
                output=1,
                reasoning=1,
                cache=AssistantMessageTokensCache(read=1, write=0),
            ),
            finish="stop",
        )

    parts: list[Any] = []

    if msg.text:
        parts.append(
            TextPart(
                id=ulid_factory(),
                sessionID=session_id,
                messageID=msg_id,
                type="text",
                text=msg.text,
                time=TextPartTime(start=created, end=created),
            )
        )

    if msg.attached_sources:
        for source in msg.attached_sources:
            parts.append(
                ToolPart(
                    id=ulid_factory(),
                    sessionID=session_id,
                    messageID=msg_id,
                    type="tool",
                    callID=ulid_factory(),
                    tool="search_knowledge_base",
                    state=ToolStateCompleted(
                        status="completed",
                        input={"query": "retrieval"},
                        output=str(source),
                        title="Knowledge Base Search",
                        metadata=source if isinstance(source, dict) else {},
                        time=ToolStateCompletedTime(start=created, end=created),
                    ),
                )
            )

    return {"info": info, "parts": parts}


# =============================================================================
# STATIC DATA GENERATORS
# =============================================================================


def get_default_provider() -> dict[str, Any]:
    """single litellm provider with a default model."""
    provider = Provider(
        id=DEFAULT_PROVIDER_ID,
        name="LiteLLM",
        source="config",
        env=[],
        options={},
        models={
            DEFAULT_MODEL_ID: Model(
                id=DEFAULT_MODEL_ID,
                providerID=DEFAULT_PROVIDER_ID,
                api=ModelApi(id="litellm", url="", npm=""),
                name="Default Model",
                capabilities=ModelCapabilities(
                    temperature=True,
                    reasoning=False,
                    attachment=False,
                    toolcall=True,
                    input=ModelCapabilitiesInput(
                        text=True, audio=False, image=False, video=False, pdf=False
                    ),
                    output=ModelCapabilitiesOutput(
                        text=True, audio=False, image=False, video=False, pdf=False
                    ),
                    interleaved=False,
                ),
                cost=ModelCost(
                    input=0,
                    output=0,
                    cache=ModelCostCache(read=0, write=0),
                    experimentalOver200K=ModelCostExperimentalOver200k(
                        input=0,
                        output=0,
                        cache=ModelCostExperimentalOver200kCache(read=0, write=0),
                    ),
                ),
                limit=ModelLimit(context=128000, output=4096),
                status="active",
                options={},
                headers={},
                releaseDate="2025-01-01",
            )
        },
    )
    return provider.model_dump()


def get_agents() -> list[dict[str, Any]]:
    return [
        Agent(
            name="fosra",
            description="Knowledge retrieval assistant",
            mode="primary",
            permission=PermissionRuleset(),
            options={},
            topP=None,
        ).model_dump(),
        Agent(
            name="fosra-coding",
            description="Coding assistant with file access",
            mode="primary",
            color="#00ff88",
            permission=PermissionRuleset(),
            options={},
            topP=None,
        ).model_dump(),
    ]


def get_default_config() -> dict[str, Any]:
    return {
        "theme": "dark",
        "autoPair": True,
    }


# =============================================================================
# CORE: SESSION
# =============================================================================

SessionTimeUpdated = Literal["updated"]
SessionTimeArchiving = Literal["archiving"]
SessionTimeCompacting = Literal["compacting"]
SessionTimeCompact = Literal["compact"]
SessionTimeActive = Literal["active"]


class SessionTime(BaseModel):
    created: int
    updated: int
    compacting: Optional[int] = None
    archived: Optional[int] = None


class SessionSummary(BaseModel):
    additions: int
    deletions: int
    files: int
    diffs: Optional[List[FileDiff]] = None


class SessionShare(BaseModel):
    url: str


class SessionRevert(BaseModel):
    messageID: str
    partID: Optional[str] = None
    snapshot: Optional[str] = None
    diff: Optional[str] = None


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
    permission: Optional[PermissionRuleset] = None
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
    permission: Optional[PermissionRuleset] = None
    revert: Optional[SessionRevert] = None
    project: Optional[ProjectSummary] = None


# =============================================================================
# CORE: MESSAGE
# =============================================================================

OutputFormatText = Literal["text"]


class OutputFormat(BaseModel):
    type: OutputFormatText


class UserMessageTime(BaseModel):
    created: int


class UserMessageModel(BaseModel):
    providerID: str
    modelID: str


class UserMessageSummary(BaseModel):
    title: Optional[str] = None
    body: Optional[str] = None
    diffs: List[FileDiff] = Field(default_factory=list)


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
    error: Optional[
        Union[
            ProviderAuthError,
            UnknownError,
            MessageOutputLengthError,
            MessageAbortedError,
            StructuredOutputError,
            ContextOverflowError,
            ApiError,
        ]
    ] = None
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


Message = Union[UserMessage, AssistantMessage]


# =============================================================================
# CORE: PART TYPES
# =============================================================================

TextPartTimeStartEnd = Literal["start", "end"]


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
    error: ApiError
    time: RetryPartTime


class CompactionPart(BaseModel):
    id: str
    sessionID: str
    messageID: str
    type: Literal["compaction"]
    auto: bool
    overflow: Optional[bool] = None


Part = Union[
    TextPart,
    ToolPart,
    ReasoningPart,
    FilePart,
    StepStartPart,
    StepFinishPart,
    AgentPart,
    SubtaskPart,
    RetryPart,
    CompactionPart,
    SnapshotPart,
    PatchPart,
]


# =============================================================================
# CORE: TOOL STATE
# =============================================================================

ToolStateStatus = Literal["pending", "running", "completed", "error"]


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


ToolState = Union[
    ToolStatePending, ToolStateRunning, ToolStateCompleted, ToolStateError
]


class ToolPart(BaseModel):
    id: str
    sessionID: str
    messageID: str
    type: Literal["tool"]
    callID: str
    tool: str
    state: ToolState
    metadata: Optional[Dict[str, Any]] = None


# =============================================================================
# CORE: SESSION STATUS
# =============================================================================

SessionStatusType = Literal["idle", "busy", "retry"]


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


# =============================================================================
# CORE: ERROR TYPES
# =============================================================================

class ApiErrorData(BaseModel):
    message: str
    statusCode: Optional[int] = None
    isRetryable: bool
    responseHeaders: Optional[Dict[str, str]] = None
    responseBody: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None


class ApiError(BaseModel):
    name: Literal["APIError"]
    data: ApiErrorData


class ProviderAuthErrorData(BaseModel):
    providerID: str
    message: str


class ProviderAuthError(BaseModel):
    name: Literal["ProviderAuthError"]
    data: ProviderAuthErrorData


class UnknownErrorData(BaseModel):
    message: str


class UnknownError(BaseModel):
    name: Literal["UnknownError"]
    data: UnknownErrorData


class MessageOutputLengthErrorData(BaseModel):
    pass


class MessageOutputLengthError(BaseModel):
    name: Literal["MessageOutputLengthError"]
    data: MessageOutputLengthErrorData


class MessageAbortedErrorData(BaseModel):
    message: str


class MessageAbortedError(BaseModel):
    name: Literal["MessageAbortedError"]
    data: MessageAbortedErrorData


class StructuredOutputErrorData(BaseModel):
    message: str
    retries: int


class StructuredOutputError(BaseModel):
    name: Literal["StructuredOutputError"]
    data: StructuredOutputErrorData


class ContextOverflowErrorData(BaseModel):
    message: str
    responseBody: Optional[str] = None


class ContextOverflowError(BaseModel):
    name: Literal["ContextOverflowError"]
    data: ContextOverflowErrorData


# =============================================================================
# PERMISSION & QUESTION
# =============================================================================

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


# =============================================================================
# TODO
# =============================================================================

class Todo(BaseModel):
    content: str
    status: str
    priority: str


# =============================================================================
# TUI EVENTS (server → TUI via SSE)
# =============================================================================

class EventTuiPromptAppendProperties(BaseModel):
    text: str


class EventTuiPromptAppend(BaseModel):
    type: Literal["tui.prompt.append"]
    properties: EventTuiPromptAppendProperties


class EventTuiCommandExecuteProperties(BaseModel):
    command: Union[
        Literal[
            "session.list",
            "session.new",
            "session.share",
            "session.interrupt",
            "session.compact",
            "session.page.up",
            "session.page.down",
            "session.line.up",
            "session.line.down",
            "session.half.page.up",
            "session.half.page.down",
            "session.first",
            "session.last",
            "prompt.clear",
            "prompt.submit",
            "agent.cycle",
        ],
        str,
    ]


class EventTuiCommandExecute(BaseModel):
    type: Literal["tui.command.execute"]
    properties: EventTuiCommandExecuteProperties


class EventTuiToastShowProperties(BaseModel):
    title: Optional[str] = None
    message: str
    variant: Literal["info", "success", "warning", "error"]
    duration: Optional[int] = None


class EventTuiToastShow(BaseModel):
    type: Literal["tui.toast.show"]
    properties: EventTuiToastShowProperties


class EventTuiSessionSelectProperties(BaseModel):
    sessionID: str


class EventTuiSessionSelect(BaseModel):
    type: Literal["tui.session.select"]
    properties: EventTuiSessionSelectProperties


# =============================================================================
# MESSAGE EVENTS
# =============================================================================

class EventMessageUpdatedProperties(BaseModel):
    info: Message


class EventMessageUpdated(BaseModel):
    type: Literal["message.updated"]
    properties: EventMessageUpdatedProperties


class EventMessageRemovedProperties(BaseModel):
    sessionID: str
    messageID: str


class EventMessageRemoved(BaseModel):
    type: Literal["message.removed"]
    properties: EventMessageRemovedProperties


class EventMessagePartUpdatedProperties(BaseModel):
    part: Part


class EventMessagePartUpdated(BaseModel):
    type: Literal["message.part.updated"]
    properties: EventMessagePartUpdatedProperties


class EventMessagePartDeltaProperties(BaseModel):
    sessionID: str
    messageID: str
    partID: str
    field: str
    delta: str


class EventMessagePartDelta(BaseModel):
    type: Literal["message.part.delta"]
    properties: EventMessagePartDeltaProperties


class EventMessagePartRemovedProperties(BaseModel):
    sessionID: str
    messageID: str
    partID: str


class EventMessagePartRemoved(BaseModel):
    type: Literal["message.part.removed"]
    properties: EventMessagePartRemovedProperties


# =============================================================================
# SESSION EVENTS
# =============================================================================

class EventSessionCreatedProperties(BaseModel):
    info: Session


class EventSessionCreated(BaseModel):
    type: Literal["session.created"]
    properties: EventSessionCreatedProperties


class EventSessionUpdatedProperties(BaseModel):
    info: Session


class EventSessionUpdated(BaseModel):
    type: Literal["session.updated"]
    properties: EventSessionUpdatedProperties


class EventSessionDeletedProperties(BaseModel):
    info: Session


class EventSessionDeleted(BaseModel):
    type: Literal["session.deleted"]
    properties: EventSessionDeletedProperties


class EventSessionStatusProperties(BaseModel):
    sessionID: str
    status: SessionStatus


class EventSessionStatus(BaseModel):
    type: Literal["session.status"]
    properties: EventSessionStatusProperties


class EventSessionIdleProperties(BaseModel):
    sessionID: str


class EventSessionIdle(BaseModel):
    type: Literal["session.idle"]
    properties: EventSessionIdleProperties


class EventSessionCompactedProperties(BaseModel):
    sessionID: str


class EventSessionCompacted(BaseModel):
    type: Literal["session.compacted"]
    properties: EventSessionCompactedProperties


class EventSessionDiffProperties(BaseModel):
    sessionID: str
    diff: List[FileDiff]


class EventSessionDiff(BaseModel):
    type: Literal["session.diff"]
    properties: EventSessionDiffProperties


class EventSessionErrorProperties(BaseModel):
    sessionID: Optional[str] = None
    error: Optional[
        Union[
            ProviderAuthError,
            UnknownError,
            MessageOutputLengthError,
            MessageAbortedError,
            StructuredOutputError,
            ContextOverflowError,
            ApiError,
        ]
    ] = None


class EventSessionError(BaseModel):
    type: Literal["session.error"]
    properties: EventSessionErrorProperties


# =============================================================================
# PERMISSION & QUESTION EVENTS
# =============================================================================

class EventPermissionAsked(BaseModel):
    type: Literal["permission.asked"]
    properties: PermissionRequest


class EventPermissionRepliedProperties(BaseModel):
    sessionID: str
    requestID: str
    reply: Literal["once", "always", "reject"]


class EventPermissionReplied(BaseModel):
    type: Literal["permission.replied"]
    properties: EventPermissionRepliedProperties


class EventQuestionAsked(BaseModel):
    type: Literal["question.asked"]
    properties: QuestionRequest


class EventQuestionRepliedProperties(BaseModel):
    sessionID: str
    requestID: str
    answers: List[List[str]]


class EventQuestionReplied(BaseModel):
    type: Literal["question.replied"]
    properties: EventQuestionRepliedProperties


class EventQuestionRejectedProperties(BaseModel):
    sessionID: str
    requestID: str


class EventQuestionRejected(BaseModel):
    type: Literal["question.rejected"]
    properties: EventQuestionRejectedProperties


# =============================================================================
# TODO EVENTS
# =============================================================================

class EventTodoUpdatedProperties(BaseModel):
    sessionID: str
    todos: List[Todo]


class EventTodoUpdated(BaseModel):
    type: Literal["todo.updated"]
    properties: EventTodoUpdatedProperties


# =============================================================================
# VCS / PATH EVENTS
# =============================================================================

class EventVcsBranchUpdatedProperties(BaseModel):
    branch: Optional[str] = None


class EventVcsBranchUpdated(BaseModel):
    type: Literal["vcs.branch.updated"]
    properties: EventVcsBranchUpdatedProperties


class EventLspUpdated(BaseModel):
    type: Literal["lsp.updated"]
    properties: Dict[str, Any]


# =============================================================================
# PROVIDER / AGENT / MODEL (bootstrap data)
# =============================================================================

class ModelApi(BaseModel):
    id: str
    url: str
    npm: str


class ModelCapabilitiesInput(BaseModel):
    text: bool
    audio: bool
    image: bool
    video: bool
    pdf: bool


class ModelCapabilitiesOutput(BaseModel):
    text: bool
    audio: bool
    image: bool
    video: bool
    pdf: bool


class ModelCapabilitiesInterleaved(BaseModel):
    field: Literal["reasoning_content", "reasoning_details"]


class ModelCapabilities(BaseModel):
    temperature: bool
    reasoning: bool
    attachment: bool
    toolcall: bool
    input: ModelCapabilitiesInput
    output: ModelCapabilitiesOutput
    interleaved: Union[bool, ModelCapabilitiesInterleaved]


class ModelCostCache(BaseModel):
    read: float
    write: float


class ModelCostExperimentalOver200kCache(BaseModel):
    read: float
    write: float


class ModelCostExperimentalOver200k(BaseModel):
    input: float
    output: float
    cache: ModelCostExperimentalOver200kCache


class ModelCost(BaseModel):
    input: float
    output: float
    cache: ModelCostCache
    experimentalOver200K: Optional[ModelCostExperimentalOver200k] = None


class ModelLimit(BaseModel):
    context: int
    input: Optional[int] = None
    output: int


class Model(BaseModel):
    id: str
    providerID: str
    api: ModelApi
    name: str
    family: Optional[str] = None
    capabilities: ModelCapabilities
    cost: ModelCost
    limit: ModelLimit
    status: Literal["alpha", "beta", "deprecated", "active"]
    options: Dict[str, Any]
    headers: Dict[str, Any]
    releaseDate: str
    variants: Optional[Dict[str, Dict[str, Any]]] = None


class Provider(BaseModel):
    id: str
    name: str
    source: Literal["env", "config", "custom", "api"]
    env: List[str]
    key: Optional[str] = None
    options: Dict[str, Any]
    models: Dict[str, Model]


class AgentModel(BaseModel):
    modelID: str
    providerID: str


class Agent(BaseModel):
    name: str
    description: Optional[str] = None
    mode: Literal["subagent", "primary", "all"]
    native: Optional[bool] = None
    hidden: Optional[bool] = None
    topP: Optional[float] = None
    temperature: Optional[float] = None
    color: Optional[str] = None
    permission: PermissionRuleset
    model: Optional[AgentModel] = None
    variant: Optional[str] = None
    prompt: Optional[str] = None
    options: Dict[str, Any]
    steps: Optional[int] = None


# =============================================================================
# CONFIG
# =============================================================================

class LogLevel(BaseModel):
    pass


class ServerConfig(BaseModel):
    port: Optional[int] = None
    hostname: Optional[str] = None
    mdns: Optional[bool] = None
    mdnsDomain: Optional[str] = None
    cors: Optional[List[str]] = None


class ConfigCommandEntry(BaseModel):
    template: str
    description: Optional[str] = None
    agent: Optional[str] = None
    model: Optional[str] = None
    subtask: Optional[bool] = None


class ConfigSkills(BaseModel):
    paths: Optional[List[str]] = None
    urls: Optional[List[str]] = None


class ConfigWatcher(BaseModel):
    ignore: Optional[List[str]] = None


class PermissionConfig(BaseModel):
    pass


class ConfigCompaction(BaseModel):
    auto: Optional[bool] = None
    prune: Optional[bool] = None
    reserved: Optional[int] = None


class ConfigExperimental(BaseModel):
    disablePasteSummary: Optional[bool] = None
    batchTool: Optional[bool] = None
    openTelemetry: Optional[bool] = None
    primaryTools: Optional[List[str]] = None
    continueLoopOnDeny: Optional[bool] = None
    mcpTimeout: Optional[int] = None


class ConfigEnterprise(BaseModel):
    url: Optional[str] = None


class AgentConfig(BaseModel):
    model: Optional[str] = None
    variant: Optional[str] = None
    temperature: Optional[float] = None
    topP: Optional[float] = None
    prompt: Optional[str] = None
    tools: Optional[Dict[str, bool]] = None
    disable: Optional[bool] = None
    description: Optional[str] = None
    mode: Optional[Literal["subagent", "primary", "all"]] = None
    hidden: Optional[bool] = None
    options: Optional[Dict[str, Any]] = None
    color: Optional[str] = None
    steps: Optional[int] = None
    maxSteps: Optional[int] = None
    permission: Optional[PermissionConfig] = None


class ConfigAgent(BaseModel):
    plan: Optional[AgentConfig] = None
    build: Optional[AgentConfig] = None
    general: Optional[AgentConfig] = None
    explore: Optional[AgentConfig] = None
    title: Optional[AgentConfig] = None
    summary: Optional[AgentConfig] = None
    compaction: Optional[AgentConfig] = None


class ProviderConfigOptions(BaseModel):
    apiKey: Optional[str] = None
    baseURL: Optional[str] = None
    enterpriseUrl: Optional[str] = None
    setCacheKey: Optional[bool] = None
    timeout: Optional[Union[int, bool]] = None
    chunkTimeout: Optional[int] = None


class ProviderConfigModelProvider(BaseModel):
    npm: Optional[str] = None
    api: Optional[str] = None


class ProviderConfigModelModalities(BaseModel):
    input: List[Literal["text", "audio", "image", "video", "pdf"]]
    output: List[Literal["text", "audio", "image", "video", "pdf"]]


class ProviderConfigModelLimit(BaseModel):
    context: int
    input: Optional[int] = None
    output: int


class ProviderConfigModelCostContextOver200k(BaseModel):
    input: float
    output: float
    cacheRead: Optional[float] = None
    cacheWrite: Optional[float] = None


class ProviderConfigModelCost(BaseModel):
    input: float
    output: float
    cacheRead: Optional[float] = None
    cacheWrite: Optional[float] = None
    contextOver200k: Optional[ProviderConfigModelCostContextOver200k] = None


class ProviderConfigModelInterleaved(BaseModel):
    field: Literal["reasoning_content", "reasoning_details"]


class ProviderConfigModelVariant(BaseModel):
    disabled: Optional[bool] = None


class ProviderConfigModel(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    family: Optional[str] = None
    releaseDate: Optional[str] = None
    attachment: Optional[bool] = None
    reasoning: Optional[bool] = None
    temperature: Optional[bool] = None
    toolCall: Optional[bool] = None
    interleaved: Optional[Union[bool, ProviderConfigModelInterleaved]] = None
    cost: Optional[ProviderConfigModelCost] = None
    limit: Optional[ProviderConfigModelLimit] = None
    modalities: Optional[ProviderConfigModelModalities] = None
    experimental: Optional[bool] = None
    status: Optional[Literal["alpha", "beta", "deprecated"]] = None
    options: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, str]] = None
    provider: Optional[ProviderConfigModelProvider] = None
    variants: Optional[Dict[str, ProviderConfigModelVariant]] = None


class ProviderConfig(BaseModel):
    api: Optional[str] = None
    name: Optional[str] = None
    env: Optional[List[str]] = None
    id: Optional[str] = None
    npm: Optional[str] = None
    models: Optional[Dict[str, ProviderConfigModel]] = None
    whitelist: Optional[List[str]] = None
    blacklist: Optional[List[str]] = None
    options: Optional[ProviderConfigOptions] = None


class McpLocalConfig(BaseModel):
    type: Literal["local"]
    command: List[str]
    environment: Optional[Dict[str, str]] = None
    enabled: Optional[bool] = None
    timeout: Optional[int] = None


class McpRemoteConfig(BaseModel):
    type: Literal["remote"]
    url: str
    enabled: Optional[bool] = None
    headers: Optional[Dict[str, str]] = None
    oauth: Optional[Union[McpOAuthConfig, bool]] = None
    timeout: Optional[int] = None


class McpOAuthConfig(BaseModel):
    clientId: Optional[str] = None
    clientSecret: Optional[str] = None
    scope: Optional[str] = None


class ConfigMcpValue(BaseModel):
    enabled: Optional[bool] = None


class ConfigFormatter(BaseModel):
    pass


class ConfigLsp(BaseModel):
    pass


class LayoutConfig(BaseModel):
    pass


class Config(BaseModel):
    $schema: Optional[str] = None
    logLevel: Optional[LogLevel] = None
    server: Optional[ServerConfig] = None
    command: Optional[Dict[str, ConfigCommandEntry]] = None
    skills: Optional[ConfigSkills] = None
    watcher: Optional[ConfigWatcher] = None
    plugin: Optional[List[str]] = None
    snapshot: Optional[bool] = None
    share: Optional[Literal["manual", "auto", "disabled"]] = None
    autoshare: Optional[bool] = None
    autoupdate: Optional[Union[bool, Literal["notify"]]] = None
    disabledProviders: Optional[List[str]] = None
    enabledProviders: Optional[List[str]] = None
    model: Optional[str] = None
    smallModel: Optional[str] = None
    defaultAgent: Optional[str] = None
    username: Optional[str] = None
    mode: Optional[Dict[str, Optional[AgentConfig]]] = None
    agent: Optional[ConfigAgent] = None
    provider: Optional[Dict[str, ProviderConfig]] = None
    mcp: Optional[Dict[str, Union[McpLocalConfig, McpRemoteConfig, ConfigMcpValue]]] = None
    formatter: Optional[Union[bool, ConfigFormatter]] = None
    lsp: Optional[Union[bool, ConfigLsp]] = None
    instructions: Optional[List[str]] = None
    layout: Optional[LayoutConfig] = None
    permission: Optional[PermissionConfig] = None
    tools: Optional[Dict[str, bool]] = None
    enterprise: Optional[ConfigEnterprise] = None
    compaction: Optional[ConfigCompaction] = None
    experimental: Optional[ConfigExperimental] = None


# =============================================================================
# PROMPT REQUEST (inbound from TUI)
# =============================================================================

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


class PromptRequest(BaseModel):
    sessionID: str
    messageID: Optional[str] = None
    parts: List[
        Union[TextPartInput, FilePartInput, AgentPartInput, SubtaskPartInput]
    ] = Field(default_factory=list)
    model: Optional[AgentModel] = None
    agent: Optional[str] = None
    variant: Optional[str] = None


# =============================================================================
# FILE / VCS / PATH
# =============================================================================

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


class FileNode(BaseModel):
    name: str
    path: str
    absolute: str
    type: Literal["file", "directory"]
    ignored: bool


class FileContentPatchHunk(BaseModel):
    oldStart: int
    oldLines: int
    newStart: int
    newLines: int
    lines: List[str]


class FileContentPatch(BaseModel):
    oldFileName: str
    newFileName: str
    oldHeader: Optional[str] = None
    newHeader: Optional[str] = None
    hunks: List[FileContentPatchHunk]
    index: Optional[str] = None


class FileContent(BaseModel):
    type: Literal["text", "binary"]
    content: str
    diff: Optional[str] = None
    patch: Optional[FileContentPatch] = None
    encoding: Optional[Literal["base64"]] = None
    mimeType: Optional[str] = None


class File(BaseModel):
    path: str
    added: int
    removed: int
    status: Literal["added", "deleted", "modified"]


class VcsInfo(BaseModel):
    branch: str


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


# =============================================================================
# LSP / FORMATTER STATUS
# =============================================================================

class LspStatus(BaseModel):
    id: str
    name: str
    root: str
    status: Literal["connected", "error"]


class FormatterStatus(BaseModel):
    name: str
    extensions: List[str]
    enabled: bool


# =============================================================================
# MCP
# =============================================================================

class McpResource(BaseModel):
    name: str
    uri: str
    description: Optional[str] = None
    mimeType: Optional[str] = None
    client: str


class McpStatusConnected(BaseModel):
    status: Literal["connected"]


class McpStatusDisabled(BaseModel):
    status: Literal["disabled"]


class McpStatusFailed(BaseModel):
    status: Literal["failed"]
    error: str


class McpStatusNeedsAuth(BaseModel):
    status: Literal["needs_auth"]


class McpStatusNeedsClientRegistration(BaseModel):
    status: Literal["needs_client_registration"]
    error: str


class McpStatus(BaseModel):
    pass


class ToolIds(BaseModel):
    pass


class ToolListItem(BaseModel):
    id: str
    description: str
    parameters: Any


class ToolList(BaseModel):
    pass


class EventMcpToolsChangedProperties(BaseModel):
    server: str


class EventMcpToolsChanged(BaseModel):
    type: Literal["mcp.tools.changed"]
    properties: EventMcpToolsChangedProperties


# =============================================================================
# PROVIDER AUTH
# =============================================================================

class ProviderAuthMethodPromptWhen(BaseModel):
    key: str
    op: Literal["eq", "neq"]
    value: str


class ProviderAuthMethodPromptSelectOption(BaseModel):
    label: str
    value: str
    hint: Optional[str] = None


class ProviderAuthMethodPromptSelect(BaseModel):
    type: Literal["select"]
    key: str
    message: str
    options: List[ProviderAuthMethodPromptSelectOption]
    when: Optional[ProviderAuthMethodPromptWhen] = None


class ProviderAuthMethodPromptText(BaseModel):
    type: Literal["text"]
    key: str
    message: str
    placeholder: Optional[str] = None
    when: Optional[ProviderAuthMethodPromptWhen] = None


class ProviderAuthMethodPrompt(BaseModel):
    pass


class ProviderAuthMethod(BaseModel):
    type: Literal["oauth", "api"]
    label: str
    prompts: Optional[
        List[Union[ProviderAuthMethodPromptText, ProviderAuthMethodPromptSelect]]
    ] = None


class ProviderAuthAuthorization(BaseModel):
    url: str
    method: Literal["auto", "code"]
    instructions: str


# =============================================================================
# WORKSPACE / PROJECT
# =============================================================================

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


class Workspace(BaseModel):
    id: str
    type: str
    branch: Optional[str] = None
    name: Optional[str] = None
    directory: Optional[str] = None
    extra: Optional[Any] = None
    projectID: str


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


class EventProjectUpdated(BaseModel):
    type: Literal["project.updated"]
    properties: Project


class EventWorkspaceReadyProperties(BaseModel):
    name: str


class EventWorkspaceReady(BaseModel):
    type: Literal["workspace.ready"]
    properties: EventWorkspaceReadyProperties


class EventWorkspaceFailedProperties(BaseModel):
    message: str


class EventWorkspaceFailed(BaseModel):
    type: Literal["workspace.failed"]
    properties: EventWorkspaceFailedProperties


# =============================================================================
# SYMBOL / CODE NAVIGATION
# =============================================================================

class SymbolLocation(BaseModel):
    uri: str
    range: Range


class Symbol(BaseModel):
    name: str
    kind: int
    location: SymbolLocation


# =============================================================================
# AUTH TYPES
# =============================================================================

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


# =============================================================================
# ERROR RESPONSES
# =============================================================================

class NotFoundErrorData(BaseModel):
    message: str


class NotFoundError(BaseModel):
    name: Literal["NotFoundError"]
    data: NotFoundErrorData


class BadRequestError(BaseModel):
    data: Any
    errors: List[Dict[str, Any]]
    success: Literal[False]


# =============================================================================
# PTY (pseudo-terminal - stub, not used in FOSRA)
# =============================================================================

class Pty(BaseModel):
    id: str
    title: str
    command: str
    args: List[str]
    cwd: str
    status: Literal["running", "exited"]
    pid: int


class EventPtyCreatedProperties(BaseModel):
    info: Pty


class EventPtyCreated(BaseModel):
    type: Literal["pty.created"]
    properties: EventPtyCreatedProperties


class EventPtyUpdatedProperties(BaseModel):
    info: Pty


class EventPtyUpdated(BaseModel):
    type: Literal["pty.updated"]
    properties: EventPtyUpdatedProperties


class EventPtyExitedProperties(BaseModel):
    id: str
    exitCode: int


class EventPtyExited(BaseModel):
    type: Literal["pty.exited"]
    properties: EventPtyExitedProperties


class EventPtyDeletedProperties(BaseModel):
    id: str


class EventPtyDeleted(BaseModel):
    type: Literal["pty.deleted"]
    properties: EventPtyDeletedProperties


# =============================================================================
# GLOBAL / MULTI-INSTANCE (stub - not used in FOSRA)
# =============================================================================

class ClientOptions(BaseModel):
    baseUrl: str


class GlobalEvent(BaseModel):
    directory: str
    payload: Event


class EventServerConnected(BaseModel):
    type: Literal["server.connected"]
    properties: Dict[str, Any]


class EventServerInstanceDisposedProperties(BaseModel):
    directory: str


class EventServerInstanceDisposed(BaseModel):
    type: Literal["server.instance.disposed"]
    properties: EventServerInstanceDisposedProperties


class EventGlobalDisposedProperties(BaseModel):
    pass


class EventGlobalDisposed(BaseModel):
    type: Literal["global.disposed"]
    properties: EventGlobalDisposedProperties


# =============================================================================
# FILE WATCHER (stub - not used in FOSRA)
# =============================================================================

class EventFileEditedProperties(BaseModel):
    file: str


class EventFileEdited(BaseModel):
    type: Literal["file.edited"]
    properties: EventFileEditedProperties


class EventFileWatcherUpdatedProperties(BaseModel):
    file: str
    event: Literal["add", "change", "unlink"]


class EventFileWatcherUpdated(BaseModel):
    type: Literal["file.watcher.updated"]
    properties: EventFileWatcherUpdatedProperties


# =============================================================================
# LSP DIAGNOSTICS (stub - not used in FOSRA)
# =============================================================================

class EventLspClientDiagnosticsProperties(BaseModel):
    serverID: str
    path: str


class EventLspClientDiagnostics(BaseModel):
    type: Literal["lsp.client.diagnostics"]
    properties: EventLspClientDiagnosticsProperties


# =============================================================================
# INSTALLATION / UPDATE (stub - not used in FOSRA)
# =============================================================================

class EventInstallationUpdatedProperties(BaseModel):
    version: str


class EventInstallationUpdated(BaseModel):
    type: Literal["installation.updated"]
    properties: EventInstallationUpdatedProperties


class EventInstallationUpdateAvailableProperties(BaseModel):
    version: str


class EventInstallationUpdateAvailable(BaseModel):
    type: Literal["installation.update-available"]
    properties: EventInstallationUpdateAvailableProperties


# =============================================================================
# MCP BROWSER (stub - not used in FOSRA)
# =============================================================================

class EventMcpBrowserOpenFailedProperties(BaseModel):
    mcpName: str
    url: str


class EventMcpBrowserOpenFailed(BaseModel):
    type: Literal["mcp.browser.open.failed"]
    properties: EventMcpBrowserOpenFailedProperties


# =============================================================================
# COMMAND EXECUTED (stub - not used in FOSRA)
# =============================================================================

class EventCommandExecutedProperties(BaseModel):
    name: str
    sessionID: str
    arguments: str
    messageID: str


class EventCommandExecuted(BaseModel):
    type: Literal["command.executed"]
    properties: EventCommandExecutedProperties


# =============================================================================
# WORKTREE EVENTS (stub - not used in FOSRA)
# =============================================================================

class EventWorktreeReadyProperties(BaseModel):
    name: str
    branch: str


class EventWorktreeReady(BaseModel):
    type: Literal["worktree.ready"]
    properties: EventWorktreeReadyProperties


class EventWorktreeFailedProperties(BaseModel):
    message: str


class EventWorktreeFailed(BaseModel):
    type: Literal["worktree.failed"]
    properties: EventWorktreeFailedProperties


# =============================================================================
# UNION TYPES (must be after all members are defined)
# =============================================================================

Event = Union[
    # TUI
    EventTuiPromptAppend,
    EventTuiCommandExecute,
    EventTuiToastShow,
    EventTuiSessionSelect,
    # Message
    EventMessageUpdated,
    EventMessageRemoved,
    EventMessagePartUpdated,
    EventMessagePartDelta,
    EventMessagePartRemoved,
    # Session
    EventSessionCreated,
    EventSessionUpdated,
    EventSessionDeleted,
    EventSessionStatus,
    EventSessionIdle,
    EventSessionCompacted,
    EventSessionDiff,
    EventSessionError,
    # Permission / Question
    EventPermissionAsked,
    EventPermissionReplied,
    EventQuestionAsked,
    EventQuestionReplied,
    EventQuestionRejected,
    # Todo
    EventTodoUpdated,
    # VCS / LSP
    EventVcsBranchUpdated,
    EventLspUpdated,
    # MCP
    EventMcpToolsChanged,
    # Project / Workspace
    EventProjectUpdated,
    EventWorkspaceReady,
    EventWorkspaceFailed,
    # Server
    EventServerConnected,
    EventServerInstanceDisposed,
    EventGlobalDisposed,
    # File
    EventFileEdited,
    EventFileWatcherUpdated,
    # LSP Diagnostics
    EventLspClientDiagnostics,
    # Installation
    EventInstallationUpdated,
    EventInstallationUpdateAvailable,
    # MCP Browser
    EventMcpBrowserOpenFailed,
    # Command
    EventCommandExecuted,
    # PTY
    EventPtyCreated,
    EventPtyUpdated,
    EventPtyExited,
    EventPtyDeleted,
    # Worktree
    EventWorktreeReady,
    EventWorktreeFailed,
]


# =============================================================================
# REBUILD FORWARD REFERENCES
# =============================================================================

EventMessageUpdatedProperties.model_rebuild()
EventMessagePartUpdatedProperties.model_rebuild()
Session.model_rebuild()
GlobalSession.model_rebuild()
EventSessionCreatedProperties.model_rebuild()
EventSessionUpdatedProperties.model_rebuild()
EventSessionDeletedProperties.model_rebuild()
EventSessionErrorProperties.model_rebuild()
ProviderAuthMethodPrompt.model_rebuild()
EventTuiCommandExecuteProperties.model_rebuild()


# =============================================================================
# IMPORTS FROM API SCHEMAS
# =============================================================================

from backend.src.api.schemas.api_schemas import (
    ConvoFullResponse,
    ConvoListItemResponse,
    MessageResponse,
)

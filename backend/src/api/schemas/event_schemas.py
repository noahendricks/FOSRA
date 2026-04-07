"""event schemas — all SSE event types (server → TUI)."""

from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel

try:
    from backend.src.api.schemas.config_schemas import (  # noqa: F401
        ApiError,
        ContextOverflowError,
        MessageAbortedError,
        MessageOutputLengthError,
        ProviderAuthError,
        StructuredOutputError,
        UnknownError,
    )

    SessionErrorType = (
        ApiError
        | ProviderAuthError
        | UnknownError
        | MessageOutputLengthError
        | MessageAbortedError
        | StructuredOutputError
        | ContextOverflowError
    )
except ImportError:
    SessionErrorType = Any

# ---- TUI EVENTS ----


class EventTuiPromptAppendProperties(BaseModel):
    text: str


class EventTuiPromptAppend(BaseModel):
    type: Literal["tui.prompt.append"]
    properties: EventTuiPromptAppendProperties


class EventTuiCommandExecuteProperties(BaseModel):
    command: (
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
        ]
        | str
    )


class EventTuiCommandExecute(BaseModel):
    type: Literal["tui.command.execute"]
    properties: EventTuiCommandExecuteProperties


class EventTuiToastShowProperties(BaseModel):
    title: str | None = None
    message: str
    variant: Literal["info", "success", "warning", "error"]
    duration: int | None = None


class EventTuiToastShow(BaseModel):
    type: Literal["tui.toast.show"]
    properties: EventTuiToastShowProperties


class EventTuiSessionSelectProperties(BaseModel):
    sessionID: str


class EventTuiSessionSelect(BaseModel):
    type: Literal["tui.session.select"]
    properties: EventTuiSessionSelectProperties


# ---- MESSAGE EVENTS ----


class EventMessageUpdatedProperties(BaseModel):
    info: Any  # Message — avoid circular


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
    part: Any  # Part — avoid circular


class EventMessagePartUpdated(BaseModel):
    type: Literal["message.part.updated"]
    properties: EventMessagePartUpdatedProperties


class EventMessagePartDeltaProperties(BaseModel):
    sessionID: str
    messageID: str
    partID: str
    field: str
    delta: str
    partType: str = "text"


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


# ---- SESSION EVENTS ----


class EventSessionCreatedProperties(BaseModel):
    info: Any  # Session — avoid circular


class EventSessionCreated(BaseModel):
    type: Literal["session.created"]
    properties: EventSessionCreatedProperties


class EventSessionUpdatedProperties(BaseModel):
    info: Any  # Session — avoid circular


class EventSessionUpdated(BaseModel):
    type: Literal["session.updated"]
    properties: EventSessionUpdatedProperties


class EventSessionDeletedProperties(BaseModel):
    info: Any  # Session — avoid circular


class EventSessionDeleted(BaseModel):
    type: Literal["session.deleted"]
    properties: EventSessionDeletedProperties


class EventSessionStatusProperties(BaseModel):
    sessionID: str
    status: Any  # SessionStatus — avoid circular


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
    diff: list[Any]  # FileDiff — avoid circular


class EventSessionDiff(BaseModel):
    type: Literal["session.diff"]
    properties: EventSessionDiffProperties


class EventSessionErrorProperties(BaseModel):
    sessionID: str | None = None
    error: Union[
        "ApiError",
        "ProviderAuthError",
        "UnknownError",
        "MessageOutputLengthError",
        "MessageAbortedError",
        "StructuredOutputError",
        "ContextOverflowError",
        None,
    ] = None


class EventSessionError(BaseModel):
    type: Literal["session.error"]
    properties: EventSessionErrorProperties


# ---- PERMISSION & QUESTION EVENTS ----


class EventPermissionAsked(BaseModel):
    type: Literal["permission.asked"]
    properties: Any  # PermissionRequest — avoid circular


class EventPermissionRepliedProperties(BaseModel):
    sessionID: str
    requestID: str
    reply: Literal["once", "always", "reject"]


class EventPermissionReplied(BaseModel):
    type: Literal["permission.replied"]
    properties: EventPermissionRepliedProperties


class EventQuestionAsked(BaseModel):
    type: Literal["question.asked"]
    properties: Any  # QuestionRequest — avoid circular


class EventQuestionRepliedProperties(BaseModel):
    sessionID: str
    requestID: str
    answers: list[list[str]]


class EventQuestionReplied(BaseModel):
    type: Literal["question.replied"]
    properties: EventQuestionRepliedProperties


class EventQuestionRejectedProperties(BaseModel):
    sessionID: str
    requestID: str


class EventQuestionRejected(BaseModel):
    type: Literal["question.rejected"]
    properties: EventQuestionRejectedProperties


# ---- TODO EVENTS ----


class EventTodoCreatedProperties(BaseModel):
    sessionID: str
    todo: Any  # Todo — avoid circular


class EventTodoCreated(BaseModel):
    type: Literal["todo.created"]
    properties: EventTodoCreatedProperties


class EventTodoUpdatedProperties(BaseModel):
    sessionID: str
    todos: list[dict[str, Any]]


class EventTodoUpdated(BaseModel):
    type: Literal["todo.updated"]
    properties: EventTodoUpdatedProperties


class EventTodoDeletedProperties(BaseModel):
    sessionID: str
    todo: Any  # Todo — avoid circular


class EventTodoDeleted(BaseModel):
    type: Literal["todo.deleted"]
    properties: EventTodoDeletedProperties


# ---- VCS / LSP EVENTS ----


class EventVcsBranchUpdatedProperties(BaseModel):
    branch: str | None = None


class EventVcsBranchUpdated(BaseModel):
    type: Literal["vcs.branch.updated"]
    properties: EventVcsBranchUpdatedProperties


class EventLspUpdated(BaseModel):
    type: Literal["lsp.updated"]
    properties: dict[str, Any]


# ---- MCP EVENTS ----


class EventMcpToolsChangedProperties(BaseModel):
    server: str


class EventMcpToolsChanged(BaseModel):
    type: Literal["mcp.tools.changed"]
    properties: EventMcpToolsChangedProperties


# ---- SERVER / GLOBAL EVENTS ----


class EventServerConnected(BaseModel):
    type: Literal["server.connected"]
    properties: dict[str, Any] = {}


class EventServerInstanceDisposedProperties(BaseModel):
    directory: str


class EventServerInstanceDisposed(BaseModel):
    type: Literal["server.instance.disposed"]
    properties: EventServerInstanceDisposedProperties


class EventGlobalDisposedProperties(BaseModel):
    directory: str


class EventGlobalDisposed(BaseModel):
    type: Literal["global.disposed"]
    properties: EventGlobalDisposedProperties


# ---- FILE EVENTS ----


class EventFileEditedProperties(BaseModel):
    file: str


class EventFileEdited(BaseModel):
    type: Literal["file.edited"]
    properties: EventFileEditedProperties


class EventFileWatcherUpdatedProperties(BaseModel):
    pass


class EventFileWatcherUpdated(BaseModel):
    type: Literal["file.watcher.updated"]
    properties: EventFileWatcherUpdatedProperties


# ---- LSP DIAGNOSTICS ----


class EventLspClientDiagnosticsProperties(BaseModel):
    uri: str
    diagnostics: list[Any]


class EventLspClientDiagnostics(BaseModel):
    type: Literal["lsp.client.diagnostics"]
    properties: EventLspClientDiagnosticsProperties


# ---- INSTALLATION EVENTS ----


class EventInstallationUpdatedProperties(BaseModel):
    version: str


class EventInstallationUpdated(BaseModel):
    type: Literal["installation.updated"]
    properties: EventInstallationUpdatedProperties


class EventInstallationUpdateAvailableProperties(BaseModel):
    currentVersion: str
    latestVersion: str


class EventInstallationUpdateAvailable(BaseModel):
    type: Literal["installation.update-available"]
    properties: EventInstallationUpdateAvailableProperties


# ---- MCP BROWSER ----


class EventMcpBrowserOpenFailedProperties(BaseModel):
    url: str


class EventMcpBrowserOpenFailed(BaseModel):
    type: Literal["mcp.browser.open.failed"]
    properties: EventMcpBrowserOpenFailedProperties


# ---- COMMAND / WORKTREE EVENTS ----


class EventCommandExecutedProperties(BaseModel):
    id: str
    exitCode: int
    output: str | None = None


class EventCommandExecuted(BaseModel):
    type: Literal["command.executed"]
    properties: EventCommandExecutedProperties


# ---- PROJECT / WORKSPACE EVENTS ----


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


class ProjectProperties(BaseModel):
    id: str
    worktree: str
    vcs: Literal["git"] | None = None
    name: str | None = None
    icon: ProjectIcon | None = None
    commands: ProjectCommands | None = None
    time: ProjectTime
    sandboxes: list[str]


class EventProjectUpdated(BaseModel):
    type: Literal["project.updated"]
    properties: ProjectProperties


# ---- GLOBAL EVENT ENVELOPE ----


class GlobalEvent(BaseModel):
    pass

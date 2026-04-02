"""event schemas — all SSE event types (server → TUI)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel

if TYPE_CHECKING:
    pass

# ---- TUI EVENTS ----


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
    diff: List[Any]  # FileDiff — avoid circular


class EventSessionDiff(BaseModel):
    type: Literal["session.diff"]
    properties: EventSessionDiffProperties


class EventSessionErrorProperties(BaseModel):
    sessionID: Optional[str] = None
    error: Optional[Any] = None  # ApiError/ProviderAuthError — avoid circular


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


# ---- TODO EVENTS ----


class EventTodoCreatedProperties(BaseModel):
    sessionID: str
    todo: Any  # Todo — avoid circular


class EventTodoCreated(BaseModel):
    type: Literal["todo.created"]
    properties: EventTodoCreatedProperties


class EventTodoUpdatedProperties(BaseModel):
    sessionID: str
    todo: Any  # Todo — avoid circular


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
    branch: Optional[str] = None


class EventVcsBranchUpdated(BaseModel):
    type: Literal["vcs.branch.updated"]
    properties: EventVcsBranchUpdatedProperties


class EventLspUpdated(BaseModel):
    type: Literal["lsp.updated"]
    properties: Dict[str, Any]


# ---- MCP EVENTS ----


class EventMcpToolsChangedProperties(BaseModel):
    server: str


class EventMcpToolsChanged(BaseModel):
    type: Literal["mcp.tools.changed"]
    properties: EventMcpToolsChangedProperties


# ---- PTY EVENTS ----


class EventPtyCreatedProperties(BaseModel):
    id: str
    cwd: str
    cols: int
    rows: int


class EventPtyCreated(BaseModel):
    type: Literal["pty.created"]
    properties: EventPtyCreatedProperties


class EventPtyUpdatedProperties(BaseModel):
    id: str
    data: Optional[str] = None
    cols: Optional[int] = None
    rows: Optional[int] = None
    title: Optional[str] = None


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


# ---- SERVER / GLOBAL EVENTS ----


class EventServerConnected(BaseModel):
    type: Literal["server.connected"]
    properties: Dict[str, Any] = {}


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
    path: str


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
    diagnostics: List[Any]


class EventLspClientDiagnostics(BaseModel):
    type: Literal["lsp.client.diagnostics"]
    properties: EventLspClientDiagnosticsProperties


# ---- INSTALLATION EVENTS ----


class EventInstallationUpdatedProperties(BaseModel):
    version: str
    releaseNotes: str


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
    output: Optional[str] = None


class EventCommandExecuted(BaseModel):
    type: Literal["command.executed"]
    properties: EventCommandExecutedProperties


class EventWorktreeReadyProperties(BaseModel):
    name: str
    path: str


class EventWorktreeReady(BaseModel):
    type: Literal["worktree.ready"]
    properties: EventWorktreeReadyProperties


class EventWorktreeFailedProperties(BaseModel):
    name: str
    error: str


class EventWorktreeFailed(BaseModel):
    type: Literal["worktree.failed"]
    properties: EventWorktreeFailedProperties


# ---- PROJECT / WORKSPACE EVENTS ----


class EventProjectUpdated(BaseModel):
    type: Literal["project.updated"]
    properties: Dict[str, Any]


class EventWorkspaceReadyProperties(BaseModel):
    workspaceID: str
    directory: str


class EventWorkspaceReady(BaseModel):
    type: Literal["workspace.ready"]
    properties: EventWorkspaceReadyProperties


class EventWorkspaceFailedProperties(BaseModel):
    workspaceID: str
    error: str


class EventWorkspaceFailed(BaseModel):
    type: Literal["workspace.failed"]
    properties: EventWorkspaceFailedProperties


# ---- GLOBAL EVENT ENVELOPE ----


class GlobalEvent(BaseModel):
    pass

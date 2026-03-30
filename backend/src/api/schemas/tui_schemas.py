"""
tui-compatible types and fosra → tui shape transformers.

maps fosra domain objects (convos, messages) into the shapes the
solidjs tui expects for sessions, messages, parts, and events.

All schema classes have been moved to focused modules:
- session_schemas: Session, SessionTime, SessionSummary, SessionStatus, etc.
- message_schemas: UserMessage, AssistantMessage, all Part types, ToolState, etc.
- event_schemas: all Event* types
- config_schemas: Provider, Model, Agent, ServerConfig, Config, error types
- tui_control_schemas: PromptRequest, FileDiff, PermissionRequest, QuestionRequest, Todo, Path, Command, etc.
- source_schemas: FileNode, FileContent, VcsInfo, LspStatus, ToolList, etc.
- workspace_schemas: Workspace, Project, Worktree, Auth, Symbol, error responses, Pty

This file re-exports everything for backward compatibility.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any


# =============================================================================
# CONSTANTS
# =============================================================================

PROJECT_DIR = os.environ.get("FOSRA_PROJECT_DIR", os.getcwd())
DEFAULT_USER_ID = os.environ.get("FOSRA_USER_ID", "dev-user000")
DEFAULT_PROVIDER_ID = os.environ.get("FOSRA_PROVIDER_ID", "openai")
DEFAULT_MODEL_ID = os.environ.get("FOSRA_MODEL_ID", "gpt-4o")

# =============================================================================
# RE-EXPORTS FROM FOCUSED SCHEMAS (backward compatibility)
# =============================================================================

from backend.src.api.schemas.session_schemas import (  # noqa: F401
    GlobalSession,
    Session,
    SessionRevert,
    SessionShare,
    SessionStatus,
    SessionStatusBusy,
    SessionStatusIdle,
    SessionStatusRetry,
    SessionSummary,
    SessionTime,
)
from backend.src.api.schemas.message_schemas import (  # noqa: F401
    AgentPart,
    AgentPartSource,
    AssistantMessage,
    AssistantMessagePath,
    AssistantMessageTime,
    AssistantMessageTokens,
    AssistantMessageTokensCache,
    CompactionPart,
    Message,
    OutputFormat,
    Part,
    PatchPart,
    ReasoningPart,
    ReasoningPartTime,
    RetryPart,
    RetryPartTime,
    SnapshotPart,
    StepFinishPart,
    StepFinishPartTokens,
    StepFinishPartTokensCache,
    StepStartPart,
    SubtaskPart,
    SubtaskPartModel,
    TextPart,
    TextPartTime,
    ToolPart,
    ToolState,
    ToolStateCompleted,
    ToolStateCompletedTime,
    ToolStateError,
    ToolStateErrorTime,
    ToolStatePending,
    ToolStateRunning,
    ToolStateRunningTime,
    UserMessage,
    UserMessageModel,
    UserMessageSummary,
    UserMessageTime,
)
from backend.src.api.schemas.config_schemas import (  # noqa: F401
    Agent,
    AgentConfig,
    ApiError,
    ApiErrorData,
    Config,
    ConfigAgent,
    ConfigCompaction,
    ConfigEnterprise,
    ConfigExperimental,
    ConfigFormatter,
    ConfigLsp,
    ConfigMcpValue,
    ConfigSkills,
    ConfigWatcher,
    ContextOverflowError,
    ContextOverflowErrorData,
    LogLevel,
    McpLocalConfig,
    McpOAuthConfig,
    McpRemoteConfig,
    MessageAbortedError,
    MessageAbortedErrorData,
    MessageOutputLengthError,
    MessageOutputLengthErrorData,
    Model,
    ModelApi,
    ModelCapabilities,
    ModelCapabilitiesInput,
    ModelCapabilitiesInterleaved,
    ModelCapabilitiesOutput,
    ModelCost,
    ModelCostCache,
    ModelCostExperimentalOver200k,
    ModelCostExperimentalOver200kCache,
    ModelLimit,
    PermissionConfig,
    Provider,
    ProviderAuthError,
    ProviderAuthErrorData,
    ProviderConfig,
    ProviderConfigModel,
    ProviderConfigModelCost,
    ProviderConfigModelCostContextOver200k,
    ProviderConfigModelInterleaved,
    ProviderConfigModelLimit,
    ProviderConfigModelModalities,
    ProviderConfigModelProvider,
    ProviderConfigModelVariant,
    ProviderConfigOptions,
    ServerConfig,
    StructuredOutputError,
    StructuredOutputErrorData,
    UnknownError,
    UnknownErrorData,
    AgentModel,
    ConfigCommandEntry,
    LayoutConfig,
)
from backend.src.api.schemas.tui_control_schemas import (  # noqa: F401,F811
    AgentPartInput,
    AgentPartInputSource,
    AgentModel,
    Command,
    FileDiff,
    FilePartInput,
    FilePartSource,
    FilePartSourceText,
    Path,
    PermissionAction,
    PermissionRequest,
    PermissionRequestTool,
    PermissionRule,
    PermissionRuleset,
    PromptRequest,
    QuestionAnswer,
    QuestionInfo,
    QuestionOption,
    QuestionRequest,
    QuestionRequestTool,
    Range,
    RangeEnd,
    RangeStart,
    ResourceSource,
    SubtaskPartInput,
    SubtaskPartModel,
    SymbolSource,
    TextPartInput,
    Todo,
)
from backend.src.api.schemas.source_schemas import (  # noqa: F401,F811
    File,
    FileContent,
    FileContentPatch,
    FileContentPatchHunk,
    FileNode,
    FormatterStatus,
    LspStatus,
    McpResource,
    ToolIds,
    ToolList,
    ToolListItem,
    VcsInfo,
    McpStatus,
)
from backend.src.api.schemas.event_schemas import (  # noqa: F401
    EventFileEdited,
    EventFileEditedProperties,
    EventFileWatcherUpdated,
    EventFileWatcherUpdatedProperties,
    EventGlobalDisposed,
    EventGlobalDisposedProperties,
    EventInstallationUpdateAvailable,
    EventInstallationUpdateAvailableProperties,
    EventInstallationUpdated,
    EventInstallationUpdatedProperties,
    EventLspClientDiagnostics,
    EventLspClientDiagnosticsProperties,
    EventLspUpdated,
    EventMessagePartDelta,
    EventMessagePartDeltaProperties,
    EventMessagePartRemoved,
    EventMessagePartRemovedProperties,
    EventMessagePartUpdated,
    EventMessagePartUpdatedProperties,
    EventMessageRemoved,
    EventMessageRemovedProperties,
    EventMessageUpdated,
    EventMessageUpdatedProperties,
    EventMcpBrowserOpenFailed,
    EventMcpBrowserOpenFailedProperties,
    EventMcpToolsChanged,
    EventMcpToolsChangedProperties,
    EventPermissionAsked,
    EventPermissionReplied,
    EventPermissionRepliedProperties,
    EventProjectUpdated,
    EventPtyCreated,
    EventPtyCreatedProperties,
    EventPtyDeleted,
    EventPtyDeletedProperties,
    EventPtyExited,
    EventPtyExitedProperties,
    EventPtyUpdated,
    EventPtyUpdatedProperties,
    EventQuestionAsked,
    EventQuestionRejected,
    EventQuestionRejectedProperties,
    EventQuestionReplied,
    EventQuestionRepliedProperties,
    EventSessionCompacted,
    EventSessionCompactedProperties,
    EventSessionCreated,
    EventSessionCreatedProperties,
    EventSessionDeleted,
    EventSessionDeletedProperties,
    EventSessionDiff,
    EventSessionDiffProperties,
    EventSessionError,
    EventSessionErrorProperties,
    EventSessionIdle,
    EventSessionIdleProperties,
    EventSessionStatus,
    EventSessionStatusProperties,
    EventSessionUpdated,
    EventSessionUpdatedProperties,
    EventTodoUpdated,
    EventTodoUpdatedProperties,
    EventTuiCommandExecute,
    EventTuiCommandExecuteProperties,
    EventTuiPromptAppend,
    EventTuiPromptAppendProperties,
    EventTuiSessionSelect,
    EventTuiSessionSelectProperties,
    EventTuiToastShow,
    EventTuiToastShowProperties,
    EventVcsBranchUpdated,
    EventVcsBranchUpdatedProperties,
    EventWorktreeFailed,
    EventWorktreeFailedProperties,
    EventWorktreeReady,
    EventWorktreeReadyProperties,
    EventWorkspaceFailed,
    EventWorkspaceFailedProperties,
    EventWorkspaceReady,
    EventWorkspaceReadyProperties,
    GlobalEvent,
    EventServerConnected,
    EventServerInstanceDisposed,
    EventServerInstanceDisposedProperties,
)
from backend.src.api.schemas.workspace_schemas import (  # noqa: F401,F811
    ApiAuth,
    Auth,
    BadRequestError,
    NotFoundError,
    NotFoundErrorData,
    OAuth,
    Project,
    ProjectCommands,
    ProjectIcon,
    ProjectSummary,
    ProjectTime,
    Pty,
    Symbol,
    SymbolLocation,
    Range,
    RangeEnd,
    RangeStart,
    WellKnownAuth,
    Worktree,
    WorktreeCreateInput,
    WorktreeRemoveInput,
    WorktreeResetInput,
    Workspace,
)
from backend.src.api.schemas.convo_api_schemas import (  # noqa: F401
    ConvoListItemResponse,
    ConvoFullResponse,
)
from backend.src.api.schemas.api_schemas import (  # noqa: F401
    MessageResponse,
)

# =============================================================================
# UTILITY FUNCTIONS (remain in this file)
# =============================================================================


def _ts(dt: datetime | None) -> int:
    return int(dt.timestamp()) if dt else 0


def convo_to_session(
    item: ConvoListItemResponse,
) -> Session:
    return Session(
        id=item.convo_id,
        slug="",
        projectID="",
        workspaceID="default",
        directory=PROJECT_DIR,
        title=item.title or "",
        version="",
        time=SessionTime(
            created=_ts(item.created_at),
            updated=_ts(item.updated_at),
        ),
    )


def convo_list_item_to_session(item: ConvoListItemResponse) -> Session:
    return convo_to_session(item)


def convo_full_to_session(convo: ConvoFullResponse) -> Session:
    return Session(
        id=convo.convo_id,
        slug="",
        projectID="",
        workspaceID="default",
        directory=PROJECT_DIR,
        title=convo.title or "",
        version="",
        time=SessionTime(
            created=_ts(convo.created_at),
            updated=_ts(convo.updated_at),
        ),
    )


def message_to_tui(msg: MessageResponse, session_id: str) -> dict[str, Any]:
    """convert a db MessageResponse into the {info: Message, parts: Part[]} shape the tui expects."""
    if not hasattr(msg, "model_dump"):
        return {}

    ts = int(msg.timestamp.timestamp()) if msg.timestamp else 0
    msg_id = msg.message_id or ""
    role_val = msg.role.value if hasattr(msg.role, "value") else str(msg.role)

    if role_val == "user":
        info = UserMessage(
            id=msg_id,
            sessionID=session_id,
            role="user",
            time=UserMessageTime(created=ts),
            agent="default",
            model=UserMessageModel(
                providerID=DEFAULT_PROVIDER_ID,
                modelID=DEFAULT_MODEL_ID,
            ),
        )
    else:
        info = AssistantMessage(
            id=msg_id,
            sessionID=session_id,
            role="assistant",
            time=AssistantMessageTime(created=ts, completed=ts),
            parentID=msg.parent_id or "",
            modelID=DEFAULT_MODEL_ID,
            providerID=DEFAULT_PROVIDER_ID,
            mode="default",
            agent="default",
            path=AssistantMessagePath(cwd=PROJECT_DIR, root=PROJECT_DIR),
            cost=0,
            tokens=AssistantMessageTokens(
                input=0,
                output=0,
                reasoning=0,
                cache=AssistantMessageTokensCache(read=0, write=0),
            ),
        )

    parts: list[dict[str, Any]] = []
    if msg.text:
        parts.append(
            TextPart(
                id=f"{msg_id}-text",
                sessionID=session_id,
                messageID=msg_id,
                type="text",
                text=msg.text,
            ).model_dump(mode="json")
        )

    return {"info": info.model_dump(mode="json"), "parts": parts}


def get_default_provider() -> dict[str, Any]:
    return {"providerID": DEFAULT_PROVIDER_ID, "modelID": DEFAULT_MODEL_ID}


def get_agents() -> list[dict[str, Any]]:
    return []


def get_default_config() -> dict[str, Any]:
    return {
        "model": DEFAULT_MODEL_ID,
        "providerID": DEFAULT_PROVIDER_ID,
    }

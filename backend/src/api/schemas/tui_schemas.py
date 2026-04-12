"""
tui-compatible types and fosra → tui shape transformers.

maps fosra domain objects (sessions, messages) into the shapes the
solidjs tui expects for sessions, messages, parts, and events.

all schema classes have been moved to focused modules:
- session_schemas: Session, SessionTime, SessionSummary, SessionStatus, etc.
- message_schemas: UserMessage, AssistantMessage, all Part types, ToolState, etc.
- event_schemas: all Event* types
- config_schemas: Provider, Model, Agent, ServerConfig, Config, error types
- tui_control_schemas: PromptRequest, FileDiff, PermissionRequest, QuestionRequest, Todo, Path, Command, etc.
- source_schemas: FileNode, FileContent, VcsInfo, LspStatus, ToolList, etc.
- workspace_schemas: Workspace, Project, Worktree, Auth, Symbol, error responses, Pty

this file re-exports everything for backward compatibility.
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
DEFAULT_PROVIDER_ID = os.environ.get("FOSRA_PROVIDER_ID", "ollama")
DEFAULT_MODEL_ID = os.environ.get("FOSRA_MODEL_ID", "stable-code:3b")


# =============================================================================
# RE-EXPORTS FROM FOCUSED SCHEMAS (backward compatibility)
# =============================================================================

from backend.src.api.schemas.config_schemas import (  # noqa: F401
    Agent,
    AgentConfig,
    AgentModel,
    ApiError,
    ApiErrorData,
    Config,
    ConfigAgent,
    ConfigCommandEntry,
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
    LayoutConfig,
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
    EventMcpBrowserOpenFailed,
    EventMcpBrowserOpenFailedProperties,
    EventMcpToolsChanged,
    EventMcpToolsChangedProperties,
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
    EventPermissionAsked,
    EventPermissionReplied,
    EventPermissionRepliedProperties,
    EventProjectUpdated,
    EventQuestionAsked,
    EventQuestionRejected,
    EventQuestionRejectedProperties,
    EventQuestionReplied,
    EventQuestionRepliedProperties,
    EventServerConnected,
    EventServerInstanceDisposed,
    EventServerInstanceDisposedProperties,
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
    GlobalEvent,
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
from backend.src.api.schemas.session_api_schemas import (  # noqa: F401
    SessionFullResponse,
    SessionListItemResponse,
)
from backend.src.api.schemas.session_schemas import (  # noqa: F401
    GlobalSession,
    Session,
    SessionRevert,
    SessionStatus,
    SessionStatusBusy,
    SessionStatusIdle,
    SessionStatusRetry,
    SessionTime,
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
    McpStatus,
    ToolIds,
    ToolList,
    ToolListItem,
    VcsInfo,
)
from backend.src.api.schemas.tui_control_schemas import (  # noqa: F401,F811
    AgentModel,
    AgentPartInput,
    AgentPartInputSource,
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
    QuestionReject,
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
    Range,
    RangeEnd,
    RangeStart,
    Symbol,
    SymbolLocation,
    WellKnownAuth,
    Workspace,
    Worktree,
    WorktreeCreateInput,
    WorktreeRemoveInput,
    WorktreeResetInput,
)

# =============================================================================
# UTILITY FUNCTIONS (remain in this file)
# =============================================================================


def _ts(dt: datetime | None) -> int:
    return int(dt.timestamp()) if dt else 0


def session_to_session(
    item: SessionListItemResponse,
) -> Session:
    return Session(
        session_id=item.session_id,
        directory=item.directory or PROJECT_DIR,
        title=item.title or "",
        version=item.version or "1",
        parent_id=item.parent_id,
        time=SessionTime(
            created=_ts(item.created_at),
            updated=_ts(item.updated_at),
        ),
    )


def session_list_item_to_session(item: SessionListItemResponse) -> Session:
    return session_to_session(item)


def session_full_to_session(session: SessionFullResponse) -> Session:
    return Session(
        session_id=session.session_id,
        directory=session.directory or PROJECT_DIR,
        title=session.title or "",
        version=session.version or "1",
        parent_id=session.parent_id,
        permission=session.permission,
        revert=session.revert,
        metadata=session.metadata,
        time=SessionTime(
            created=_ts(session.created_at),
            updated=_ts(session.updated_at),
        ),
    )


def message_to_tui(
    msg: UserMessage | AssistantMessage, session_id: str
) -> dict[str, Any]:
    info: dict[str, Any] = (
        msg.model_dump(mode="json") if hasattr(msg, "model_dump") else {}
    )
    parts: list[dict[str, Any]] = []
    if isinstance(msg, AssistantMessage) and msg.parts:
        for part in msg.parts:
            parts.append(part.model_dump(mode="json"))
    return {"info": info, "parts": parts}


def get_default_provider() -> dict[str, Any]:
    return {"providerID": DEFAULT_PROVIDER_ID, "modelID": DEFAULT_MODEL_ID}


def get_agents() -> list[dict[str, Any]]:
    return [
        {
            "name": "FOSRA",
            "description": "General-purpose coding agent with codebase awareness, web search, and file editing capabilities.",
            "mode": "all",
            "native": True,
            "hidden": False,
            "color": "#22d3ee",
            "permission": [],
            "options": {},
        },
        {
            "name": "Research",
            "description": "Performs deep research on a topic using web search and fetch tools. Use for fact-checking, background research, and information gathering.",
            "mode": "subagent",
            "hidden": False,
            "permission": [],
            "options": {},
        },
        {
            "name": "Code Analysis",
            "description": "Analyzes code structure, call chains, and relationships. Use for understanding codebases, finding functions, tracing dependencies, and refactoring planning.",
            "mode": "subagent",
            "hidden": False,
            "permission": [],
            "options": {},
        },
    ]


def get_default_config() -> dict[str, Any]:
    return {
        "model": DEFAULT_MODEL_ID,
        "providerID": DEFAULT_PROVIDER_ID,
    }

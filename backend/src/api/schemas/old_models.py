from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.src.storage.utils.converters import ulid_factory

PROJECT_DIR = os.environ.get("FOSRA_PROJECT_DIR", os.getcwd())
DEFAULT_USER_ID = os.environ.get("FOSRA_USER_ID", "dev-user000")
DEFAULT_PROJECT_ID = "default"
DEFAULT_VERSION = "2"
DEFAULT_PROVIDER_ID = "litellm"
DEFAULT_MODEL_ID = "default"


class SessionInfo(BaseModel):
    id: str
    slug: str
    projectID: str = DEFAULT_PROJECT_ID
    workspaceID: str | None = None
    directory: str = PROJECT_DIR
    parentID: str | None = None
    title: str = "New Convo"
    version: str = DEFAULT_VERSION
    summary: dict[str, Any] | None = None
    share: dict[str, Any] | None = None
    time: dict[str, float]
    permission: dict[str, Any] | None = None
    revert: dict[str, Any] | None = None


# MESSAGES


class UserMessage(BaseModel):
    id: str
    sessionID: str
    role: Literal["user"] = "user"
    time: dict[str, float]
    format: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None
    agent: str = "fosra"
    model: dict[str, str] = Field(
        default_factory=lambda: {
            "providerID": DEFAULT_PROVIDER_ID,
            "modelID": DEFAULT_MODEL_ID,
        }
    )
    system: str | None = None
    tools: dict[str, bool] | None = None
    variant: str | None = None


class AssistantMessage(BaseModel):
    id: str
    sessionID: str
    role: Literal["assistant"] = "assistant"
    time: dict[str, float | None]
    error: dict[str, Any] | None = None
    parentID: str
    modelID: str = DEFAULT_MODEL_ID
    providerID: str = DEFAULT_PROVIDER_ID
    mode: str = "default"
    agent: str = "fosra"
    path: dict[str, str] = Field(
        default_factory=lambda: {"cwd": PROJECT_DIR, "root": PROJECT_DIR}
    )
    summary: bool | None = None
    cost: float = 1
    tokens: dict[str, Any] = Field(
        default_factory=lambda: {
            "input": 1,
            "output": 1,
            "reasoning": 1,
            "cache": {"read": 1, "write": 0},
        }
    )
    structured: Any | None = None
    variant: str | None = None
    finish: str | None = None


# PARTS


class TuiTextPart(BaseModel):
    id: str
    sessionID: str
    messageID: str
    type: Literal["text"] = "text"
    text: str = ""
    synthetic: bool | None = None
    ignored: bool | None = None
    time: dict[str, float | None] | None = None
    metadata: dict[str, Any] | None = None


class TuiToolPart(BaseModel):
    id: str
    sessionID: str
    messageID: str
    type: Literal["tool"] = "tool"
    callID: str
    tool: str
    state: dict[str, Any]
    metadata: dict[str, Any] | None = None


class TuiStepStartPart(BaseModel):
    id: str
    sessionID: str
    messageID: str
    type: Literal["step-start"] = "step-start"
    snapshot: str | None = None


class TuiStepFinishPart(BaseModel):
    id: str
    sessionID: str
    messageID: str
    type: Literal["step-finish"] = "step-finish"
    reason: str = "stop"
    snapshot: str | None = None
    cost: float = 1
    tokens: dict[str, Any] = Field(
        default_factory=lambda: {
            "input": 1,
            "output": 1,
            "reasoning": 1,
            "cache": {"read": 1, "write": 0},
        }
    )


# PROVIDER / AGENT / CONFIG


class TuiModel(BaseModel):
    id: str
    providerID: str
    api: dict[str, str] = Field(
        default_factory=lambda: {"id": "litellm", "url": "", "npm": ""}
    )
    name: str
    family: str | None = None
    capabilities: dict[str, Any] = Field(
        default_factory=lambda: {
            "temperature": True,
            "reasoning": False,
            "attachment": False,
            "toolcall": True,
            "input": {
                "text": True,
                "audio": False,
                "image": False,
                "video": False,
                "pdf": False,
            },
            "output": {
                "text": True,
                "audio": False,
                "image": False,
                "video": False,
                "pdf": False,
            },
            "interleaved": False,
        }
    )
    cost: dict[str, Any] = Field(
        default_factory=lambda: {
            "input": 0,
            "output": 0,
            "cache": {"read": 0, "write": 0},
        }
    )
    limit: dict[str, int] = Field(
        default_factory=lambda: {"context": 128000, "output": 4096}
    )
    status: str = "active"
    options: dict[str, Any] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)
    release_date: str = "2025-01-01"


class TuiProvider(BaseModel):
    id: str = DEFAULT_PROVIDER_ID
    name: str = "LiteLLM"
    source: str = "config"
    env: list[str] = Field(default_factory=list)
    key: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    models: dict[str, TuiModel] = Field(default_factory=dict)


class TuiAgent(BaseModel):
    name: str
    description: str | None = None
    mode: str = "primary"
    native: bool | None = None
    hidden: bool | None = None
    temperature: float | None = None
    color: str | None = None
    permission: dict[str, Any] = Field(
        default_factory=lambda: {"allow": [], "deny": []}
    )
    model: dict[str, str] | None = None
    variant: str | None = None
    prompt: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    steps: int | None = None


# PROMPT REQUEST (from tui)


class PromptRequest(BaseModel):
    sessionID: str
    messageID: str | None = None
    parts: list[dict[str, Any]] = Field(default_factory=list)
    model: dict[str, str] | None = None
    agent: str | None = None
    variant: str | None = None

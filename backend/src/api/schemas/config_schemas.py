"""config schemas — Provider, Model, Agent, ServerConfig, Config, MCP, error types."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


# ---- ERROR TYPES ----


class ApiErrorData(BaseModel):
    message: str
    statusCode: int | None = None
    isRetryable: bool
    responseHeaders: dict[str, str] | None = None
    responseBody: str | None = None
    metadata: dict[str, str] | None = None


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
    responseBody: str | None = None


class ContextOverflowError(BaseModel):
    name: Literal["ContextOverflowError"]
    data: ContextOverflowErrorData


# ---- PROVIDER / MODEL ----


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
    interleaved: bool | ModelCapabilitiesInterleaved


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
    experimentalOver200K: ModelCostExperimentalOver200k | None = None


class ModelLimit(BaseModel):
    context: int
    input: int | None = None
    output: int


class Model(BaseModel):
    id: str
    providerID: str
    api: ModelApi
    name: str
    family: str | None = None
    capabilities: ModelCapabilities
    cost: ModelCost
    limit: ModelLimit
    status: Literal["alpha", "beta", "deprecated", "active"]
    options: dict[str, Any]
    headers: dict[str, Any]
    releaseDate: str | None = None
    variants: dict[str, dict[str, Any]] | None = None


class Provider(BaseModel):
    id: str
    name: str
    source: Literal["env", "config", "custom", "api"]
    env: list[str]
    key: str | None = None
    options: dict[str, Any]
    models: dict[str, Model]


class AgentModel(BaseModel):
    modelID: str
    providerID: str


class Agent(BaseModel):
    name: str
    description: str | None = None
    mode: Literal["subagent", "primary", "all"]
    native: bool | None = None
    hidden: bool | None = None
    topP: float | None = None
    temperature: float | None = None
    color: str | None = None
    permission: Any = None  # PermissionRuleset — avoid circular
    model: AgentModel | None = None
    variant: str | None = None
    prompt: str | None = None
    options: dict[str, Any]
    steps: int | None = None


# ---- CONFIG ----


class LogLevel(BaseModel):
    pass


class ServerConfig(BaseModel):
    port: int | None = None
    hostname: str | None = None
    mdns: bool | None = None
    mdnsDomain: str | None = None
    cors: list[str] | None = None


class ConfigCommandEntry(BaseModel):
    template: str
    description: str | None = None
    agent: str | None = None
    model: str | None = None
    subtask: bool | None = None


class ConfigSkills(BaseModel):
    paths: list[str] | None = None
    urls: list[str] | None = None


class ConfigWatcher(BaseModel):
    ignore: list[str] | None = None


class PermissionConfig(BaseModel):
    pass


class ConfigCompaction(BaseModel):
    auto: bool | None = None
    prune: bool | None = None
    reserved: int | None = None


class ConfigExperimental(BaseModel):
    disablePasteSummary: bool | None = None
    batchTool: bool | None = None
    openTelemetry: bool | None = None
    primaryTools: list[str] | None = None
    continueLoopOnDeny: bool | None = None
    mcpTimeout: int | None = None


class ConfigEnterprise(BaseModel):
    url: str | None = None


class AgentConfig(BaseModel):
    model: str | None = None
    variant: str | None = None
    temperature: float | None = None
    topP: float | None = None
    prompt: str | None = None
    tools: dict[str, bool] | None = None
    disable: bool | None = None
    description: str | None = None
    mode: Literal["subagent", "primary", "all"] | None = None
    hidden: bool | None = None
    options: dict[str, Any] | None = None
    color: str | None = None
    steps: int | None = None
    maxSteps: int | None = None
    permission: PermissionConfig | None = None


class ConfigAgent(BaseModel):
    plan: AgentConfig | None = None
    build: AgentConfig | None = None
    general: AgentConfig | None = None
    explore: AgentConfig | None = None
    title: AgentConfig | None = None
    summary: AgentConfig | None = None
    compaction: AgentConfig | None = None


class ProviderConfigOptions(BaseModel):
    apiKey: str | None = None
    baseURL: str | None = None
    enterpriseUrl: str | None = None
    setCacheKey: bool | None = None
    timeout: int | bool | None = None
    chunkTimeout: int | None = None


class ProviderConfigModelProvider(BaseModel):
    npm: str | None = None
    api: str | None = None


class ProviderConfigModelModalities(BaseModel):
    input: list[Literal["text", "audio", "image", "video", "pdf"]]
    output: list[Literal["text", "audio", "image", "video", "pdf"]]


class ProviderConfigModelLimit(BaseModel):
    context: int
    input: int | None = None
    output: int


class ProviderConfigModelCostContextOver200k(BaseModel):
    input: float
    output: float
    cacheRead: float | None = None
    cacheWrite: float | None = None


class ProviderConfigModelCost(BaseModel):
    input: float
    output: float
    cacheRead: float | None = None
    cacheWrite: float | None = None
    contextOver200k: ProviderConfigModelCostContextOver200k | None = None


class ProviderConfigModelInterleaved(BaseModel):
    field: Literal["reasoning_content", "reasoning_details"]


class ProviderConfigModelVariant(BaseModel):
    disabled: bool | None = None


class ProviderConfigModel(BaseModel):
    id: str | None = None
    name: str | None = None
    family: str | None = None
    releaseDate: str | None = None
    attachment: bool | None = None
    reasoning: bool | None = None
    temperature: bool | None = None
    toolCall: bool | None = None
    interleaved: bool | ProviderConfigModelInterleaved | None = None
    cost: ProviderConfigModelCost | None = None
    limit: ProviderConfigModelLimit | None = None
    modalities: ProviderConfigModelModalities | None = None
    experimental: bool | None = None
    status: Literal["alpha", "beta", "deprecated"] | None = None
    options: dict[str, Any] | None = None
    headers: dict[str, str] | None = None
    provider: ProviderConfigModelProvider | None = None
    variants: dict[str, ProviderConfigModelVariant] | None = None


class ProviderConfig(BaseModel):
    api: str | None = None
    name: str | None = None
    env: list[str] | None = None
    id: str | None = None
    npm: str | None = None
    models: dict[str, ProviderConfigModel] | None = None
    whitelist: list[str] | None = None
    blacklist: list[str] | None = None
    options: ProviderConfigOptions | None = None


class McpLocalConfig(BaseModel):
    type: Literal["local"]
    command: list[str]
    environment: dict[str, str] | None = None
    enabled: bool | None = None
    timeout: int | None = None


class McpRemoteConfig(BaseModel):
    type: Literal["remote"]
    url: str
    enabled: bool | None = None
    headers: dict[str, str] | None = None
    oauth: "McpOAuthConfig | bool | None" = None
    timeout: int | None = None


class McpOAuthConfig(BaseModel):
    clientId: str | None = None
    clientSecret: str | None = None
    scope: str | None = None


class ConfigMcpValue(BaseModel):
    enabled: bool | None = None


class ConfigFormatter(BaseModel):
    pass


class ConfigLsp(BaseModel):
    pass


class LayoutConfig(BaseModel):
    pass


class Config(BaseModel):
    logLevel: LogLevel | None = None
    server: ServerConfig | None = None
    command: dict[str, ConfigCommandEntry] | None = None
    skills: ConfigSkills | None = None
    watcher: ConfigWatcher | None = None
    plugin: list[str] | None = None
    snapshot: bool | None = None
    share: Literal["manual", "auto", "disabled"] | None = None
    autoshare: bool | None = None
    autoupdate: bool | Literal["notify"] | None = None
    disabledProviders: list[str] | None = None
    enabledProviders: list[str] | None = None
    model: str | None = None
    smallModel: str | None = None
    defaultAgent: str | None = None
    username: str | None = None
    mode: dict[str, AgentConfig | None] | None = None
    agent: ConfigAgent | None = None
    provider: dict[str, ProviderConfig] | None = None
    mcp: dict[str, McpLocalConfig | McpRemoteConfig | ConfigMcpValue] | None = None
    formatter: bool | ConfigFormatter | None = None
    lsp: bool | ConfigLsp | None = None
    instructions: list[str] | None = None
    layout: LayoutConfig | None = None
    permission: PermissionConfig | None = None
    tools: dict[str, bool] | None = None
    enterprise: ConfigEnterprise | None = None
    compaction: ConfigCompaction | None = None
    experimental: ConfigExperimental | None = None

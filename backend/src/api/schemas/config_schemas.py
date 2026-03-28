"""config schemas — Provider, Model, Agent, ServerConfig, Config, MCP, error types."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel


# ---- ERROR TYPES ----


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
    permission: Any = None  # PermissionRuleset — avoid circular
    model: Optional[AgentModel] = None
    variant: Optional[str] = None
    prompt: Optional[str] = None
    options: Dict[str, Any]
    steps: Optional[int] = None


# ---- CONFIG ----


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
    oauth: Optional[Union["McpOAuthConfig", bool]] = None
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
    mcp: Optional[Dict[str, Union[McpLocalConfig, McpRemoteConfig, ConfigMcpValue]]] = (
        None
    )
    formatter: Optional[Union[bool, ConfigFormatter]] = None
    lsp: Optional[Union[bool, ConfigLsp]] = None
    instructions: Optional[List[str]] = None
    layout: Optional[LayoutConfig] = None
    permission: Optional[PermissionConfig] = None
    tools: Optional[Dict[str, bool]] = None
    enterprise: Optional[ConfigEnterprise] = None
    compaction: Optional[ConfigCompaction] = None
    experimental: Optional[ConfigExperimental] = None

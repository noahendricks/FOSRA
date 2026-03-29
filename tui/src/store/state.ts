import { ReactiveMap } from "@solid-primitives/map"
import { createSignal } from "solid-js"
import { createStore } from "solid-js/store"
import type {
  Session,
  Message,
  Part,
  Todo,
  Provider,
  Agent,
  Config,
  PermissionRequest,
  QuestionRequest,
  McpStatus,
  LspStatus,
  FormatterStatus,
  VcsInfo,
  Command,
  Workspace,
} from "../fosra/sdk-types"

export function createAppState() {
  const sessions = new ReactiveMap<string, Session>()
  const messages = new ReactiveMap<string, Message[]>()
  const parts = new ReactiveMap<string, Part[]>()
  const todos = new ReactiveMap<string, Todo[]>()

  const permissions = new ReactiveMap<string, PermissionRequest[]>()
  const questions = new ReactiveMap<string, QuestionRequest[]>()

  const mcp = new ReactiveMap<string, McpStatus>()

  const [systemState, setSystemState] = createStore<{
    lsp: LspStatus[]
    formatter: FormatterStatus[]
    vcs: VcsInfo | undefined
    path: { home: string; state: string; config: string; worktree: string; directory: string }
    command: Command[]
    workspaceList: Workspace[]
    providers: Provider[]
    agents: Agent[]
    config: Config | null
    directory: string
  }>({
    lsp: [],
    formatter: [],
    vcs: undefined,
    path: { home: "", state: "", config: "", worktree: "", directory: "" },
    command: [],
    workspaceList: [],
    providers: [],
    agents: [],
    config: null,
    directory: "",
  })

  const loadedSessions = new Set<string>()

  return {
    sessions,
    messages,
    parts,
    todos,
    permissions,
    questions,

    mcp,

    lsp: () => systemState.lsp,
    setLsp: (data: LspStatus[]) => setSystemState("lsp", data),
    formatter: () => systemState.formatter,
    setFormatter: (data: FormatterStatus[]) => setSystemState("formatter", data),
    vcs: () => systemState.vcs,
    setVcs: (data: VcsInfo | undefined) => setSystemState("vcs", data),
    path: () => systemState.path,
    setPath: (data: typeof systemState.path) => setSystemState("path", data),
    command: () => systemState.command,
    setCommand: (data: Command[]) => setSystemState("command", data),
    workspaceList: () => systemState.workspaceList,
    setWorkspaceList: (data: Workspace[]) => setSystemState("workspaceList", data),

    providers: () => systemState.providers,
    setProviders: (data: Provider[]) => setSystemState("providers", data),
    agents: () => systemState.agents,
    setAgents: (data: Agent[]) => setSystemState("agents", data),
    config: () => systemState.config,
    setConfig: (data: Config | null) => setSystemState("config", data),

    directory: () => systemState.directory,
    setDirectory: (data: string) => setSystemState("directory", data),

    loadedSessions,
  }
}

export type AppState = ReturnType<typeof createAppState>
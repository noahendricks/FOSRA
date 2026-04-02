import { createSignal } from "solid-js";
import { createStore, produce } from "solid-js/store";
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
} from "../fosra/sdk-types";

// REACTIVE RECORD
export interface ReactiveRecord<V> {
  get(id: string): V | undefined;
  set(id: string, value: V): void;
  delete(id: string): void;
  has(id: string): boolean;
  values(): V[];
  entries(): [string, V][];
  readonly size: number;
}

// collection store shape
type Collections = {
  sessions: Record<string, Session>;
  messages: Record<string, Message[]>;
  parts: Record<string, Part[]>;
  todos: Record<string, Todo[]>;
  permissions: Record<string, PermissionRequest[]>;
  questions: Record<string, QuestionRequest[]>;
  mcp: Record<string, McpStatus>;
};

export function createAppState() {
  const [collections, setCollections] = createStore<Collections>({
    sessions: {},
    messages: {},
    parts: {},
    todos: {},
    permissions: {},
    questions: {},
    mcp: {},
  });

  // typed wrapper that enables explicit .get()/.set()/.delete() API
  // while using createStore internally for correct reactivity under nested batch()
  function makeRecord<V>(
    read: () => Record<string, V>,
    write: (id: string, value: V) => void,
    remove: (id: string) => void,
  ): ReactiveRecord<V> {
    return {
      get: (id) => read()[id],
      set: write,
      delete: remove,
      has: (id) => read()[id] !== undefined,
      values: () => Object.values(read()),
      entries: () => Object.entries(read()),
      get size() {
        return Object.keys(read()).length;
      },
    };
  }

  const sessions = makeRecord<Session>(
    () => collections.sessions,
    (id, v) => setCollections("sessions", id, v),
    (id) =>
      setCollections(
        "sessions",
        produce((d) => {
          delete d[id];
        }),
      ),
  );

  const messages = makeRecord<Message[]>(
    () => collections.messages,
    (id, v) => setCollections("messages", id, v),
    (id) =>
      setCollections(
        "messages",
        produce((d) => {
          delete d[id];
        }),
      ),
  );

  const parts = makeRecord<Part[]>(
    () => collections.parts,
    (id, v) => setCollections("parts", id, v),
    (id) =>
      setCollections(
        "parts",
        produce((d) => {
          delete d[id];
        }),
      ),
  );

  const todos = makeRecord<Todo[]>(
    () => collections.todos,
    (id, v) => setCollections("todos", id, v),
    (id) =>
      setCollections(
        "todos",
        produce((d) => {
          delete d[id];
        }),
      ),
  );

  const permissions = makeRecord<PermissionRequest[]>(
    () => collections.permissions,
    (id, v) => setCollections("permissions", id, v),
    (id) =>
      setCollections(
        "permissions",
        produce((d) => {
          delete d[id];
        }),
      ),
  );

  const questions = makeRecord<QuestionRequest[]>(
    () => collections.questions,
    (id, v) => setCollections("questions", id, v),
    (id) =>
      setCollections(
        "questions",
        produce((d) => {
          delete d[id];
        }),
      ),
  );

  const mcp = makeRecord<McpStatus>(
    () => collections.mcp,
    (id, v) => setCollections("mcp", id, v),
    (id) =>
      setCollections(
        "mcp",
        produce((d) => {
          delete d[id];
        }),
      ),
  );

  const [systemState, setSystemState] = createStore<{
    lsp: LspStatus[];
    formatter: FormatterStatus[];
    vcs: VcsInfo | undefined;
    path: {
      home: string;
      state: string;
      config: string;
      worktree: string;
      directory: string;
    };
    command: Command[];
    workspaceList: Workspace[];
    providers: Provider[];
    agents: Agent[];
    config: Config | null;
    directory: string;
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
  });

  const loadedSessions = new Set<string>();

  const [sessionsArray, setSessionsArray] = createSignal<Session[]>([]);
  const [providerDefault, setProviderDefault] = createSignal<
    Record<string, string>
  >({});
  const [providerConnected, setProviderConnected] = createSignal<string[]>([]);

  return {
    sessions,
    sessionsArray,
    setSessionsArray,
    messages,
    parts,
    todos,
    permissions,
    questions,

    mcp,

    lsp: () => systemState.lsp,
    setLsp: (data: LspStatus[]) => setSystemState("lsp", data),
    formatter: () => systemState.formatter,
    setFormatter: (data: FormatterStatus[]) =>
      setSystemState("formatter", data),
    vcs: () => systemState.vcs,
    setVcs: (data: VcsInfo | undefined) => setSystemState("vcs", data),
    path: () => systemState.path,
    setPath: (data: typeof systemState.path) => setSystemState("path", data),
    command: () => systemState.command,
    setCommand: (data: Command[]) => setSystemState("command", data),
    workspaceList: () => systemState.workspaceList,
    setWorkspaceList: (data: Workspace[]) =>
      setSystemState("workspaceList", data),

    providers: () => systemState.providers,
    setProviders: (data: Provider[]) => setSystemState("providers", data),
    agents: () => systemState.agents,
    setAgents: (data: Agent[]) => setSystemState("agents", data),
    config: () => systemState.config,
    setConfig: (data: Config | null) => setSystemState("config", data),

    directory: () => systemState.directory,
    setDirectory: (data: string) => setSystemState("directory", data),

    loadedSessions,

    providerDefault,
    setProviderDefault,
    providerConnected,
    setProviderConnected,
  };
}

export type AppState = ReturnType<typeof createAppState>;

import { createSignal } from "solid-js";
import type { Setter } from "solid-js";
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
import { log } from "@/util/log";

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

export function snapshotState(state: ReturnType<typeof createAppState>) {
  return {
    sessions: state.sessions.size,
    messages: state.messages.size,
    parts: state.parts.size,
    todos: Object.values(state.messages.values()).reduce(
      (n, msgs) => n + ((msgs as Message[] | undefined) ?? []).length,
      0,
    ),
    mcp: state.mcp.size,
    loadedSessions: state.loadedSessions.size,
    sessionsArray: state.sessionsArray().length,
    providers: state.providers().length,
    agents: state.agents().length,
    config: state.config() !== null,
    lsp: state.lsp().length,
    formatter: state.formatter().length,
    workspaceList: state.workspaceList().length,
    command: state.command().length,
    directory: state.directory(),
    providerConnected: state.providerConnected(),
  };
}

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
    collectionName: string,
  ): ReactiveRecord<V> {
    return {
      get: (id) => read()[id],
      set: (id, value) => {
        write(id, value);
        log.store.debug("STATE_SET", {
          collection: collectionName,
          id,
          size: Object.keys(read()).length,
        });
      },
      delete: (id) => {
        remove(id);
        log.store.debug("STATE_DELETE", {
          collection: collectionName,
          id,
        });
      },
      has: (id) => read()[id] !== undefined,
      values: () => Object.values(read()),
      entries: () => Object.entries(read()),
      get size() {
        return Object.keys(read()).length;
      },
    };
  }

  function createRecords<K extends keyof Collections>(key: K): ReactiveRecord<any> {
    const read = () => collections[key];
    const write = (id: string, value: any) => setCollections(key as any, id as any, value);
    const remove = (id: string) => setCollections(key as any, produce((d: any) => { delete d[id]; }));
    return makeRecord<any>(read, write, remove, key);
  }

  const sessions = createRecords("sessions");
  const messages = createRecords("messages");
  const parts = createRecords("parts");
  const todos = createRecords("todos");
  const permissions = createRecords("permissions");
  const questions = createRecords("questions");
  const mcp = createRecords("mcp");

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
  const [version, setVersion] = createSignal(0);
  const bumpVersion = () => setVersion((v) => v + 1);

  const [sessionsArray, baseSetSessionsArray] = createSignal<Session[]>([]);
  const [providerDefault, baseSetProviderDefault] = createSignal<
    Record<string, string>
  >({});
  const [providerConnected, baseSetProviderConnected] = createSignal<string[]>(
    [],
  );

  const setSessionsArray = ((val: Session[]) => {
    baseSetSessionsArray(val);
  }) as unknown as Setter<Session[]>;

  const setProviderDefault = ((val: Record<string, string>) => {
    baseSetProviderDefault(val);
    log.store.debug("STATE_SET_SIGNAL", {
      signal: "providerDefault",
      keys: Object.keys(val),
    });
  }) as unknown as Setter<Record<string, string>>;

  const setProviderConnected = ((val: string[]) => {
    baseSetProviderConnected(val);
    log.store.debug("STATE_SET_SIGNAL", {
      signal: "providerConnected",
      count: val.length,
    });
  }) as unknown as Setter<string[]>;

  // system state setters with mutation logging
  const setLsp = (data: LspStatus[]) => {
    setSystemState("lsp", data);
    log.store.debug("STATE_SET_SYSTEM", { field: "lsp", count: data.length });
  };
  const setFormatter = (data: FormatterStatus[]) => {
    setSystemState("formatter", data);
    log.store.debug("STATE_SET_SYSTEM", {
      field: "formatter",
      count: data.length,
    });
  };
  const setVcs = (data: VcsInfo | undefined) => {
    setSystemState("vcs", data);
    log.store.debug("STATE_SET_SYSTEM", {
      field: "vcs",
      hasData: data !== undefined,
    });
  };
  const setPath = (data: typeof systemState.path) => {
    setSystemState("path", data);
    log.store.debug("STATE_SET_SYSTEM", {
      field: "path",
      keys: Object.keys(data),
    });
  };
  const setCommand = (data: Command[]) => {
    setSystemState("command", data);
    log.store.debug("STATE_SET_SYSTEM", {
      field: "command",
      count: data.length,
    });
  };
  const setWorkspaceList = (data: Workspace[]) => {
    setSystemState("workspaceList", data);
    log.store.debug("STATE_SET_SYSTEM", {
      field: "workspaceList",
      count: data.length,
    });
  };
  const setProviders = (data: Provider[]) => {
    setSystemState("providers", data);
    log.store.debug("STATE_SET_SYSTEM", {
      field: "providers",
      count: data.length,
    });
  };
  const setAgents = (data: Agent[]) => {
    setSystemState("agents", data);
    log.store.debug("STATE_SET_SYSTEM", {
      field: "agents",
      count: data.length,
    });
  };
  const setConfig = (data: Config | null) => {
    setSystemState("config", data);
    log.store.debug("STATE_SET_SYSTEM", {
      field: "config",
      hasData: data !== null,
    });
  };
  const setDirectory = (data: string) => {
    setSystemState("directory", data);
    log.store.debug("STATE_SET_SYSTEM", {
      field: "directory",
      value: data,
    });
  };

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
    setLsp,
    formatter: () => systemState.formatter,
    setFormatter,
    vcs: () => systemState.vcs,
    setVcs,
    path: () => systemState.path,
    setPath,
    command: () => systemState.command,
    setCommand,
    workspaceList: () => systemState.workspaceList,
    setWorkspaceList,

    providers: () => systemState.providers,
    setProviders,
    agents: () => systemState.agents,
    setAgents,
    config: () => systemState.config,
    setConfig,

    directory: () => systemState.directory,
    setDirectory,

    loadedSessions,
    version,
    bumpVersion,

    providerDefault,
    setProviderDefault,
    providerConnected,
    setProviderConnected,
  };
}

export type AppState = ReturnType<typeof createAppState>;

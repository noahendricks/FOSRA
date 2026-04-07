import { createEffect } from "solid-js";
import { createSimpleContext } from "./helper";
import { useApi } from "./api";
import { createAppState, snapshotState } from "../store/state";
import { createStoreActions } from "../store/actions";
import { createEventRouter } from "../events/router";
import { registerSessionHandlers } from "../events/handlers/session";
import { registerMessageHandlers } from "../events/handlers/message";
import { registerSystemHandlers } from "../events/handlers/system";
import { registerUIHandlers } from "../events/handlers/ui";
import type { EventChannel } from "../events/channel";
import { useToast } from "../ui/toast";
import { useCommandDialog } from "../component/dialog-command";
import { useRoute } from "./route";
import { usePromptRef } from "./prompt";
import type { Message, Part } from "../fosra/sdk-types";
import { parseMessages, SessionSchema, TodoSchema, ProvidersResponseSchema, AgentSchema, ConfigSchema, CommandSchema, LspStatusSchema, McpStatusSchema, FormatterStatusSchema, VcsInfoSchema, PathSchema } from "@/schemas";

import { log } from "@/util/log";

const ctx = createSimpleContext({
  name: "Store",
  init: () => {
    const api = useApi();

    const state = createAppState();

    const actions = createStoreActions({
      sessions: state.sessions,
      sessionsArray: state.sessionsArray,
      setSessionsArray: state.setSessionsArray,
      messages: state.messages,
      parts: state.parts,
      todos: state.todos,
      permission: state.permissions,
      question: state.questions,
    });

    const router = createEventRouter(api.channel);

    registerSessionHandlers(router.store, actions);
    registerMessageHandlers(router.store, actions);
    registerSystemHandlers(
      router.store,
      actions,
      () => loadInitial(api, state, actions),
      state,
    );

    loadInitial(api, state, actions);

    log.startup.info("STORE_INITIALIZED", {
      sessions: state.sessionsArray(),
      providers: state.providers,
      agents: state.agents,
    });

    const SNAPSHOT_MS =
      (parseInt(process.env.FOSRA_LOG_SNAPSHOT_INTERVAL ?? "30") || 30) * 1000;
    
    let snapshotTimer: ReturnType<typeof setInterval> | undefined;

    if (SNAPSHOT_MS > 0) {
      snapshotTimer = setInterval(() => {
        log.store.debug("STATE_SNAPSHOT", snapshotState(state));
      }, SNAPSHOT_MS);
    }

    async function sessionLoad(
      sessionID: string,
      force = false,
    ): Promise<
      | { agent?: string; model?: { providerID: string; modelID: string } }
      | undefined
    > {
      if (state.loadedSessions.has(sessionID) && !force) return;

      if (!state.sessions.has(sessionID)) {
        const res = await api.fosra.session.get({ sessionID });
        if (res.data) {
          const sessionResult = SessionSchema.safeParse(res.data);
          if (!sessionResult.success) {
            log.store.warn("SESSION_GET_INVALID", { error: sessionResult.error.format(), sessionID });
          } else {
            state.sessions.set(sessionID, sessionResult.data);
            log.session.info("SESSION_LOADED", {
              sessionID,
              totalSessions: state.sessions.size,
            });
            state.setSessionsArray([...state.sessions.values()]);
          }
        }
      }

      const [messages, todos] = await Promise.all([
        api.fosra.session.messages({ sessionID, limit: 100 }),
        api.fosra.session.todo({ sessionID }),
      ]);
      log.startup.debug("CHECKING_STARTUP_CONFIG", { messages });

      const validatedMessages = parseMessages(messages.data ?? []);
      for (const msg of validatedMessages) {
        actions.setMessage(msg.info.sessionID, msg.info as Message);
        if (msg.parts?.length) {
          for (const part of msg.parts) {
            actions.setPart(msg.info.id, part as Part);
          }
        }
      }
      const rawTodos = todos.data ?? [];
      const validTodos = rawTodos.filter((t: any) => {
        const r = TodoSchema.safeParse(t);
        if (!r.success) log.store.warn("TODO_INVALID", { error: r.error.format() });
        return r.success;
      });
      state.todos.set(sessionID, validTodos);
      state.loadedSessions.add(sessionID);

      return state.sessions.get(sessionID)?.metadata;
    }

    function sessionStatus(
      sessionID: string,
    ): "idle" | "working" | "compacting" {
      const session = state.sessions.get(sessionID);
      if (!session) return "idle";
      // session.status is a SessionStatus object ({ type: "idle" | "busy" | "retry" })
      if ("status" in session && session.status) {
        const st = session.status as any;
        const type = typeof st === "string" ? st : st.type;
        if (type === "idle") return "idle";
        if (type === "busy") return "working";
        if (type === "retry") return "working";
      }
      if (session.time?.compacting) return "compacting";
      const msgs = state.messages.get(sessionID) ?? [];
      const last = msgs.at(-1);
      if (!last) return "idle";
      if (last.role === "user") return "working";
      return last.time?.completed ? "idle" : "working";
    }

    return {
      state,
      actions,
      router,
      fosra: api.fosra,
      session: {
        load: sessionLoad,
        status: sessionStatus,
      },
      dispose() {
        router.dispose();
        if (snapshotTimer !== undefined) clearInterval(snapshotTimer);
      },
    };
  },
});

async function loadInitial(
  api: { fosra: any; channel: EventChannel },
  state: ReturnType<typeof createAppState>,
  actions: ReturnType<typeof createStoreActions>,
) {
  const [providersResult, agentsResult, configResult] = await Promise.all([
    api.fosra.config.providers({}, { throwOnError: true }),
    api.fosra.app.agents({}, { throwOnError: true }),
    api.fosra.config.get({}, { throwOnError: true }),
  ]);
  const provResult = ProvidersResponseSchema.safeParse(providersResult.data);
  if (!provResult.success) {
    log.store.warn("PROVIDERS_INVALID", { error: provResult.error.format() });
  }
  const providerData = provResult.success ? provResult.data : providersResult.data as any;
  state.setProviders(providerData?.providers ?? []);
  state.setProviderDefault(providerData?.default ?? {});
  state.setProviderConnected(providerData?.connected ?? []);
  const rawAgents = agentsResult.data ?? [];
  const validAgents = rawAgents.filter((a: any) => {
    const r = AgentSchema.safeParse(a);
    if (!r.success) log.store.warn("AGENT_INVALID", { error: r.error.format(), name: a?.name });
    return r.success;
  });
  state.setAgents(validAgents);
  const configValidation = ConfigSchema.safeParse(configResult.data);
  if (!configValidation.success) {
    log.store.warn("CONFIG_INVALID", { error: configValidation.error.format() });
  }
  state.setConfig(configResult.data ?? null);

  const sessions = await api.fosra.session.list({});
  for (const session of sessions.data ?? []) {
    const r = SessionSchema.safeParse(session);
    if (!r.success) {
      log.store.warn("SESSION_LIST_INVALID", { error: r.error.format(), id: session?.id });
      continue;
    }
    state.sessions.set(r.data.id, r.data);
  }
  state.setSessionsArray([...state.sessions.values()]);

  const [
    commandResult,
    lspResult,
    mcpResult,
    formatterResult,
    vcsResult,
    pathResult,
  ] = await Promise.all([
    api.fosra.command.list().catch(() => ({ data: [] })),
    api.fosra.lsp.status().catch(() => ({ data: [] })),
    api.fosra.mcp.status().catch(() => ({ data: {} })),
    api.fosra.formatter.status().catch(() => ({ data: [] })),
    api.fosra.vcs.get().catch(() => ({ data: {} })),
    api.fosra.path.get().catch(() => ({
      data: { home: "", state: "", config: "", worktree: "", directory: "" },
    })),
  ]);

  const rawCommands = commandResult.data ?? [];
  const validCommands = rawCommands.filter((c: any) => {
    const r = CommandSchema.safeParse(c);
    if (!r.success) log.store.warn("COMMAND_INVALID", { error: r.error.format(), name: c?.name });
    return r.success;
  });
  state.setCommand(validCommands);
  const rawLsp = lspResult.data ?? [];
  const validLsp = rawLsp.filter((l: any) => {
    const r = LspStatusSchema.safeParse(l);
    if (!r.success) log.store.warn("LSP_INVALID", { error: r.error.format(), tool: l?.tool });
    return r.success;
  });
  state.setLsp(validLsp);
  for (const [k, v] of Object.entries(mcpResult.data ?? {})) {
    const r = McpStatusSchema.safeParse(v);
    if (!r.success) {
      log.store.warn("MCP_INVALID", { error: r.error.format(), server: k });
    } else {
      state.mcp.set(k, r.data);
    }
  }
  const rawFormatter = formatterResult.data ?? [];
  const validFormatter = rawFormatter.filter((f: any) => {
    const r = FormatterStatusSchema.safeParse(f);
    if (!r.success) log.store.warn("FORMATTER_INVALID", { error: r.error.format(), tool: f?.tool });
    return r.success;
  });
  state.setFormatter(validFormatter);
  const vcsValidation = VcsInfoSchema.safeParse(vcsResult.data);
  if (!vcsValidation.success) {
    log.store.warn("VCS_INVALID", { error: vcsValidation.error.format() });
  }
  state.setVcs(vcsResult.data ?? undefined);
  const pathValidation = PathSchema.safeParse(pathResult.data);
  if (!pathValidation.success) {
    log.store.warn("PATH_INVALID", { error: pathValidation.error.format() });
  }
  state.setPath(
    pathResult.data ?? {
      home: "",
      state: "",
      config: "",
      worktree: "",
      directory: "",
    },
  );
}

export const StoreProvider = ctx.provider;
export const useStore = ctx.use;

export function UIHandlers() {
  const store = useStore();
  const toast = useToast();
  const command = useCommandDialog();
  const route = useRoute();
  const promptRef = usePromptRef();

  registerUIHandlers(store.router.ui, {
    toast,
    commands: command,
    prompt: {
      append: (text: string) => {
        const ref = promptRef.current;
        if (ref?.current) {
          ref.set({
            ...ref.current,
            input: (ref.current.input ?? "") + text,
          });
        }
      },
    },
    navigate: (sessionID: string) =>
      route.navigate({ type: "session", sessionID }),
    route: { data: route.data, navigate: route.navigate },
  });

  return null;
}

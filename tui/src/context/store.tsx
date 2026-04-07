import { createEffect } from "solid-js";
import { createSimpleContext } from "./helper";
import { useApi } from "./api";
import { createAppState, snapshotState } from "../store/state";
import { createStoreActions } from "../store/actions";
import { createEventRouter } from "../events/router";
import { registerSessionHandlers } from "../events/handlers/session";
import { registerMessageHandlers } from "../events/handlers/message";
import { registerSystemHandlers } from "../events/handlers/system";
import type { EventChannel } from "../events/channel";
import { useToast } from "../ui/toast";
import { useCommandDialog } from "../component/dialog-command";
import { useRoute } from "./route";
import { usePromptRef } from "./prompt";
import type { Message, Part } from "../fosra/sdk-types";

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
          state.sessions.set(sessionID, res.data);
          log.session.info("SESSION_LOADED", {
            sessionID,
            totalSessions: state.sessions.size,
          });
          state.setSessionsArray([...state.sessions.values()]);
        }
      }

      const [messages, todos] = await Promise.all([
        api.fosra.session.messages({ sessionID, limit: 100 }),
        api.fosra.session.todo({ sessionID }),
      ]);
      log.startup.debug("CHECKING_STARTUP_CONFIG", { messages });

      for (const msg of messages.data ?? []) {
        actions.setMessage(msg.info.sessionID, msg.info as Message);
        if (msg.parts?.length) {
          for (const part of msg.parts) {
            actions.setPart(msg.info.id, part as Part);
          }
        }
      }
      state.todos.set(sessionID, todos.data ?? []);
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
  const providerData = providersResult.data as {
    providers?: any[];
    default?: Record<string, string>;
    connected?: string[];
  } | null;
  state.setProviders(providerData?.providers ?? []);
  state.setProviderDefault(providerData?.default ?? {});
  state.setProviderConnected(providerData?.connected ?? []);
  state.setAgents(agentsResult.data ?? []);
  state.setConfig(configResult.data ?? null);

  const sessions = await api.fosra.session.list({});
  for (const session of sessions.data ?? []) {
    state.sessions.set(session.id, session);
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

  state.setCommand(commandResult.data ?? []);
  state.setLsp(lspResult.data ?? []);
  for (const [k, v] of Object.entries(mcpResult.data ?? {})) {
    state.mcp.set(k, v as any);
  }
  state.setFormatter(formatterResult.data ?? []);
  state.setVcs(vcsResult.data ?? undefined);
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

function UIHandlers() {
  const store = useStore();
  const toast = useToast();
  const command = useCommandDialog();
  const route = useRoute();
  const promptRef = usePromptRef();

  createEffect(() => {
    const router = store.router;

    router.ui.on("session.deleted" as any, (props: any) => {
      if (
        route.data.type === "session" &&
        route.data.sessionID === props.info.id
      ) {
        route.navigate({ type: "home" });
        toast.show({
          variant: "info",
          message: "The current session was deleted",
        });
      }
    });

    router.ui.on("session.error" as any, (props: any) => {
      const error = props.error;
      if (!error || typeof error !== "object") return;
      if (error.name === "MessageAbortedError") return;
      const data = (error as any).data;
      const message = data?.message ?? String(error);
      toast.show({ variant: "error", message, duration: 5000 });
    });

    router.ui.on("installation.update-available" as any, (props: any) => {
      toast.show({
        variant: "info",
        title: "Update Available",
        message: `OpenCode v${props.version} is available. Run 'opencode upgrade' to update manually.`,
        duration: 10000,
      });
    });

    router.ui.on("tui.toast.show" as any, (props: any) => {
      toast.show({
        variant: props.variant ?? "info",
        title: props.title,
        message: props.message,
        duration: props.duration,
      });
    });

    router.ui.on("tui.command.execute" as any, (props: any) => {
      command.trigger(props.command);
    });

    router.ui.on("tui.prompt.append" as any, (props: any) => {
      const ref = promptRef.current;
      if (ref?.current) {
        ref.set({
          ...ref.current,
          input: (ref.current.input ?? "") + props.text,
        });
      }
    });

    router.ui.on("tui.session.select" as any, (props: any) => {
      route.navigate({ type: "session", sessionID: props.sessionID });
    });
  });

  return null;
}

export { UIHandlers };

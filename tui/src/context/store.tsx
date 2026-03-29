import { createSimpleContext } from "./helper"
import { useApi } from "./api"
import { createAppState } from "../store/state"
import { createStoreActions } from "../store/actions"
import { createEventRouter } from "../events/router"
import { registerSessionHandlers } from "../events/handlers/session"
import { registerMessageHandlers } from "../events/handlers/message"
import { registerSystemHandlers } from "../events/handlers/system"
import type { EventChannel } from "../events/channel"

const ctx = createSimpleContext({
  name: "Store",
  init: () => {
    const api = useApi()

    const state = createAppState()
    const actions = createStoreActions({
      sessions: state.sessions,
      messages: state.messages,
      parts: state.parts,
      todos: state.todos,
      permission: state.permissions,
      question: state.questions,
    })

    const router = createEventRouter(api.channel)

    registerSessionHandlers(router.store, actions)
    registerMessageHandlers(router.store, actions)
    registerSystemHandlers(router.store, actions, () => loadInitial(api, state), state)

    loadInitial(api, state)

    return {
      state,
      actions,
      router,
      fosra: api.fosra,
      dispose() {
        router.dispose()
      },
    }
  },
})

async function loadInitial(
  api: { fosra: any; channel: EventChannel },
  state: ReturnType<typeof createAppState>
) {
  const [providers, providerList, agents, config] = await Promise.all([
    api.fosra.config.providers({}, { throwOnError: true }),
    api.fosra.provider.list({}, { throwOnError: true }),
    api.fosra.app.agents({}, { throwOnError: true }),
    api.fosra.config.get({}, { throwOnError: true }),
  ])
  state.setProviders(providers.data?.providers ?? [])
  state.setAgents(agents.data ?? [])
  state.setConfig(config.data ?? null)

  const sessions = await api.fosra.session.list({})
  for (const session of sessions.data ?? []) {
    state.sessions.set(session.id, session)
  }

  const [commandResult, lspResult, mcpResult, formatterResult, vcsResult, pathResult] = await Promise.all([
    api.fosra.command.list().catch(() => ({ data: [] })),
    api.fosra.lsp.status().catch(() => ({ data: [] })),
    api.fosra.mcp.status().catch(() => ({ data: {} })),
    api.fosra.formatter.status().catch(() => ({ data: [] })),
    api.fosra.vcs.get().catch(() => ({ data: {} })),
    api.fosra.path.get().catch(() => ({ data: { home: "", state: "", config: "", worktree: "", directory: "" } })),
  ])

  state.setCommand(commandResult.data ?? [])
  state.setLsp(lspResult.data ?? [])
  for (const [k, v] of Object.entries(mcpResult.data ?? {})) {
    state.mcp.set(k, v as any)
  }
  state.setFormatter(formatterResult.data ?? [])
  state.setVcs(vcsResult.data ?? undefined)
  state.setPath(pathResult.data ?? { home: "", state: "", config: "", worktree: "", directory: "" })
}

export const StoreProvider = ctx.provider
export const useStore = ctx.use

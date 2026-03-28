import type { ApiClient } from "../api/client"
import type { AppState } from "./state"

export async function loadInitialState(
  api: ApiClient,
  state: AppState
) {
  const [providers, agents, config] = await Promise.all([
    api.fosra.config.providers({}, { throwOnError: true }),
    api.fosra.provider.list({}, { throwOnError: true }),
    api.fosra.app.agents({}, { throwOnError: true }),
    api.fosra.config.get({}, { throwOnError: true }),
  ])

  state.setProviders(providers.data?.providers ?? [])
  state.setAgents(agents.data ?? [])
  state.setConfig(config.data ?? null)

  loadSessions(api, state)
}

async function loadSessions(api: ApiClient, state: AppState) {
  const sessions = await api.fosra.session.list({})
  for (const session of sessions.data ?? []) {
    state.sessions.set(session.id, session)
  }
}

export async function loadSession(
  api: ApiClient,
  state: AppState,
  sessionId: string
) {
  if (state.loadedSessions.has(sessionId)) return

  const [messages, todos] = await Promise.all([
    api.fosra.session.messages({ sessionID: sessionId, limit: 100 }),
    api.fosra.session.todo({ sessionID: sessionId }),
  ])

  for (const msg of messages.data ?? []) {
    state.messages.set(msg.info.id, [msg.info])
    if (msg.parts) {
      state.parts.set(msg.info.id, msg.parts)
    }
  }

  state.todos.set(sessionId, todos.data ?? [])
  state.loadedSessions.add(sessionId)
}
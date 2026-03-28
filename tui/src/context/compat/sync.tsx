import { createSimpleContext } from "../helper"
import { useStore } from "../store"
import { useApi } from "../api"
import type { Session } from "../../external/sdk-types"
import type { SessionStatus } from "../../external/sdk-types"

const ctx = createSimpleContext({
  name: "SyncCompat",
  init: () => {
    const store = useStore()
    const api = useApi()

    const data = {
      get session() {
        return [...store.state.sessions.values()]
      },
      get message(): Record<string, any[]> {
        const result: Record<string, any[]> = {}
        for (const [k, v] of store.state.messages.entries()) {
          result[k] = v
        }
        return result
      },
      get part(): Record<string, any[]> {
        const result: Record<string, any[]> = {}
        for (const [k, v] of store.state.parts.entries()) {
          result[k] = v
        }
        return result
      },
      get todo(): Record<string, any[]> {
        const result: Record<string, any[]> = {}
        for (const [k, v] of store.state.todos.entries()) {
          result[k] = v
        }
        return result
      },
      get permission(): Record<string, any[]> {
        const result: Record<string, any[]> = {}
        for (const [k, v] of store.state.permissions.entries()) {
          result[k] = v
        }
        return result
      },
      get question(): Record<string, any[]> {
        const result: Record<string, any[]> = {}
        for (const [k, v] of store.state.questions.entries()) {
          result[k] = v
        }
        return result
      },
      get provider() {
        return store.state.providers()
      },
      get agent() {
        return store.state.agents()
      },
      get config() {
        return store.state.config()
      },
      provider_next: { all: [], default: {}, connected: [] },
      provider_auth: {} as Record<string, any>,
      command: [] as any[],
      session_status: {} as Record<string, any>,
      session_diff: {} as Record<string, any[]>,
      lsp: [] as any[],
      mcp: {} as Record<string, any>,
      mcp_resource: {} as Record<string, any>,
      formatter: [] as any[],
      vcs: undefined as any,
      path: { home: "", state: "", worktree: "", directory: "" } as any,
      workspaceList: [] as any[],
      provider_default: {},
      get status() {
        if (store.state.providers().length === 0) return "loading"
        return "complete"
      },
    }

    return {
      data,
      set: () => {},
      get status() {
        return data.status
      },
      get ready() {
        return data.status !== "loading"
      },
      session: {
        get(sessionID: string) {
          return store.state.sessions.get(sessionID)
        },
        status(sessionID: string) {
          const session = store.state.sessions.get(sessionID)
          if (!session) return "idle"
          if (session.time.compacting) return "compacting"
          const messages = store.state.messages.get(sessionID) ?? []
          const last = messages.at(-1)
          if (!last) return "idle"
          if (last.role === "user") return "working"
          return last.time.completed ? "idle" : "working"
        },
        async sync(sessionID: string) {
          if (store.state.loadedSessions.has(sessionID)) return
          const [messages, todos] = await Promise.all([
            api.fosra.session.messages({ sessionID, limit: 100 }),
            api.fosra.session.todo({ sessionID }),
          ])
          for (const msg of messages.data ?? []) {
            store.state.messages.set(msg.info.id, [msg.info])
            if (msg.parts) store.state.parts.set(msg.info.id, msg.parts)
          }
          store.state.todos.set(sessionID, todos.data ?? [])
          store.state.loadedSessions.add(sessionID)
        },
      },
      workspace: {
        get() { return undefined },
        sync: async () => {},
      },
    }
  },
})

export const { provider: SyncCompatProvider, use: useSyncCompat } = ctx
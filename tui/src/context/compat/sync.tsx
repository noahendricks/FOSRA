import { createEffect, on, onMount } from "solid-js"
import { createSimpleContext } from "../helper"
import { useStore } from "../store"
import { useApi } from "../api"
import { createStore } from "solid-js/store"
import { useToast } from "../../ui/toast"
import { useCommandDialog } from "../../component/dialog-command"
import { useRoute } from "../route"
import { usePromptRef } from "../prompt"

const ctx = createSimpleContext({
  name: "SyncCompat",
  init: () => {
    const store = useStore()
    const api = useApi()

    const [data, setData] = createStore({
      session: [] as any[],
      message: {} as Record<string, any[]>,
      part: {} as Record<string, any[]>,
      todo: {} as Record<string, any[]>,
      permission: {} as Record<string, any[]>,
      question: {} as Record<string, any[]>,
      provider: [] as any[],
      agent: [] as any[],
      config: null as any,
      command: [] as any[],
      lsp: [] as any[],
      mcp: {} as Record<string, any>,
      mcp_resource: {} as Record<string, any>,
      formatter: [] as any[],
      vcs: undefined as any,
      path: { home: "", state: "", worktree: "", directory: "" } as any,
      workspaceList: [] as any[],
      provider_next: { all: [], default: {}, connected: [] },
      provider_auth: {} as Record<string, any>,
      provider_default: {},
      session_status: {} as Record<string, any>,
      session_diff: {} as Record<string, any[]>,
      status: "loading" as "loading" | "partial" | "complete",
    })

    function syncData() {
      setData("session", [...store.state.sessions.values()])

      const messageMap: Record<string, any[]> = {}
      for (const [k, v] of store.state.messages.entries()) {
        messageMap[k] = v
      }
      setData("message", messageMap)

      const partMap: Record<string, any[]> = {}
      for (const [k, v] of store.state.parts.entries()) {
        partMap[k] = v
      }
      setData("part", partMap)

      const todoMap: Record<string, any[]> = {}
      for (const [k, v] of store.state.todos.entries()) {
        todoMap[k] = v
      }
      setData("todo", todoMap)

      const permMap: Record<string, any[]> = {}
      for (const [k, v] of store.state.permissions.entries()) {
        permMap[k] = v
      }
      setData("permission", permMap)

      const questMap: Record<string, any[]> = {}
      for (const [k, v] of store.state.questions.entries()) {
        questMap[k] = v
      }
      setData("question", questMap)

      setData("provider", store.state.providers())
      setData("agent", store.state.agents())
      setData("config", store.state.config())
      setData("command", store.state.command())
      setData("lsp", store.state.lsp())

      const mcpMap: Record<string, any> = {}
      for (const [k, v] of store.state.mcp.entries()) {
        mcpMap[k] = v
      }
      setData("mcp", mcpMap)

      setData("formatter", store.state.formatter())
      setData("vcs", store.state.vcs())
      setData("path", store.state.path())
      setData("workspaceList", store.state.workspaceList())

      if (store.state.providers().length === 0) {
        setData("status", "loading")
      } else {
        setData("status", "complete")
      }
    }

    const interval = setInterval(syncData, 1000)
    syncData()

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
      dispose() {
        clearInterval(interval)
      },
    }
  },
})

export const { provider: SyncCompatProvider, use: useSyncCompat } = ctx

function UIHandlers() {
  const store = useStore()
  const toast = useToast()
  const command = useCommandDialog()
  const route = useRoute()
  const promptRef = usePromptRef()

  createEffect(() => {
    const router = store.router

    router.ui.on("session.deleted" as any, (props: any) => {
      if (route.data.type === "session" && route.data.sessionID === props.info.id) {
        route.navigate({ type: "home" })
        toast.show({ variant: "info", message: "The current session was deleted" })
      }
    })

    router.ui.on("session.error" as any, (props: any) => {
      const error = props.error
      if (!error || typeof error !== "object") return
      if (error.name === "MessageAbortedError") return
      const data = (error as any).data
      const message = data?.message ?? String(error)
      toast.show({ variant: "error", message, duration: 5000 })
    })

    router.ui.on("installation.update-available" as any, (props: any) => {
      toast.show({
        variant: "info",
        title: "Update Available",
        message: `OpenCode v${props.version} is available. Run 'opencode upgrade' to update manually.`,
        duration: 10000,
      })
    })

    router.ui.on("tui.toast.show" as any, (props: any) => {
      toast.show({ variant: props.variant ?? "info", title: props.title, message: props.message, duration: props.duration })
    })

    router.ui.on("tui.command.execute" as any, (props: any) => {
      command.trigger(props.command)
    })

    router.ui.on("tui.prompt.append" as any, (props: any) => {
      const ref = promptRef.current
      if (ref?.current) {
        ref.set({
          ...ref.current,
          input: (ref.current.input ?? "") + props.text,
        })
      }
    })

    router.ui.on("tui.session.select" as any, (props: any) => {
      route.navigate({ type: "session", sessionID: props.sessionID })
    })
  })

  return null
}

export { UIHandlers }

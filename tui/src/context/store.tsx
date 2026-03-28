import { onMount } from "solid-js"
import { createSimpleContext } from "./helper"
import { useToast } from "../ui/toast"
import { useCommandDialog } from "../component/dialog-command"
import { useRoute } from "./route"
import { usePromptRef } from "./prompt"
import { createAppState } from "../store/state"
import { createStoreActions } from "../store/actions"
import { createEventRouter } from "../events/router"
import { registerSessionHandlers } from "../events/handlers/session"
import { registerMessageHandlers } from "../events/handlers/message"
import { registerSystemHandlers } from "../events/handlers/system"
import { registerUIHandlers } from "../events/handlers/ui"
import type { EventChannel } from "../events/channel"

const ctx = createSimpleContext({
  name: "Store",
  init: (props: {
    channel: EventChannel
    fosra: any
    url: string
    directory: string
    fetch: typeof fetch
  }) => {
    const route = useRoute()
    const toast = useToast()
    const command = useCommandDialog()
    const promptRef = usePromptRef()

    const state = createAppState()
    const actions = createStoreActions({
      sessions: state.sessions,
      messages: state.messages,
      parts: state.parts,
      todos: state.todos,
      permission: state.permissions,
      question: state.questions,
    })

    const router = createEventRouter(props.channel)

    registerSessionHandlers(router.store, actions)
    registerMessageHandlers(router.store, actions)
    registerSystemHandlers(router.store, actions, () => loadInitial(props, state), state)

    registerUIHandlers(router.ui, {
      toast: {
        show: (opts) => toast.show(opts as any),
      },
      commands: {
        trigger: (cmd) => command.trigger(cmd),
      },
      prompt: {
        append: (text) => {
          const ref = promptRef.current
          if (ref?.current) {
            ref.set({
              ...ref.current,
              input: (ref.current.input ?? "") + text,
            })
          }
        },
      },
      navigate: (sessionID) => {
        route.navigate({ type: "session", sessionID })
      },
    })

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

    return {
      state,
      actions,
      router,
      fosra: props.fosra,
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
}

export const StoreProvider = ctx.provider
export const useStore = ctx.use
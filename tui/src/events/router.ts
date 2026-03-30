import type { Event } from "@fosra/api/v2"
import type { EventChannel } from "./channel"
import { createHandlerRegistry } from "./types"
import { Log } from "@/util/log"

export function createEventRouter(channel: EventChannel) {
  const store = createHandlerRegistry()
  const ui = createHandlerRegistry()

  const unsubscribe = channel.subscribe((event: Event) => {
    Log.Default.debug("[SSE ROUTER] Dispatching event:", JSON.stringify(event))
    store.dispatch(event)
    ui.dispatch(event)
  })

  return {
    store,
    ui,
    dispose() {
      unsubscribe()
      channel.stop()
    },
  }
}

export type EventRouter = ReturnType<typeof createEventRouter>
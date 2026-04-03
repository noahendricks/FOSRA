import type { Event } from "@fosra/api/v2"
import type { EventChannel } from "./channel"
import { createHandlerRegistry } from "./types"
import { log } from "@/util/log"

export function createEventRouter(channel: EventChannel) {
  const store = createHandlerRegistry()
  const ui = createHandlerRegistry()

  const unsubscribe = channel.subscribe((event: Event) => {
    log.sse.debug("SSE_EVENT_DISPATCH", { event })
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
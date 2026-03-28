import type { Event } from "@fosra/sdk/v2"
import type { EventChannel } from "./channel"
import { createHandlerRegistry } from "./types"

export function createEventRouter(channel: EventChannel) {
  const store = createHandlerRegistry()
  const ui = createHandlerRegistry()

  const unsubscribe = channel.subscribe((event: Event) => {
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